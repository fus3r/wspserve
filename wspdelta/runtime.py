"""Serve several edit versions of one resident model in a single batched forward pass.

Each row of a batch carries a version id. On every edited projection the shared base
matmul runs once for all rows, each version then adds only its own typed delta to its own
rows: an input-channel mask for ChannelScale, an unmerged low-rank term for LowRank.
Rows of different versions never read each other's deltas. This is a reference
implementation for correctness checks, not an optimized kernel (see Punica/S-LoRA).
"""
from contextlib import contextmanager

import numpy as np
import torch

from .delta import ChannelScale, LowRank


def module_path(checkpoint_name, model):
    """Map a checkpoint tensor name to the nn.Linear that owns it in ``model``."""
    stem = checkpoint_name.removesuffix(".weight")
    modules = dict(model.named_modules())
    for candidate in (stem, stem.replace("model.language_model.", "model."),
                      stem.replace("model.", "model.language_model.", 1)):
        if candidate in modules:
            return candidate
    raise KeyError(checkpoint_name)


def bits_to_tensor(bits):
    return torch.from_numpy(np.ascontiguousarray(bits, dtype=np.uint16).view(np.int16)).view(torch.bfloat16)


class MultiVersion:
    """Per-row edit versions over one model. ``versions`` maps a name to {tensor name: delta}."""

    def __init__(self, model, versions, base_bits, *, factor_dtype=torch.float32, storage_dtype=None, mask=True):
        """storage_dtype (e.g. torch.bfloat16) rounds the factors as a LoRA server would store them,
        they are still applied in factor_dtype."""
        self.model = model
        self.names = ["base", *versions]
        self.ids = {name: i for i, name in enumerate(self.names)}
        self.row_versions = None
        self.handles = []
        self.scales, self.factors = {}, {}
        for version, deltas in versions.items():
            vid = self.ids[version]
            for tensor, delta in deltas.items():
                path = module_path(tensor, model)
                if isinstance(delta, ChannelScale) and mask:
                    cols = torch.as_tensor(delta.columns, dtype=torch.long)
                    scale = torch.as_tensor(delta.scales, dtype=torch.float32)
                    self.scales.setdefault(path, []).append((vid, cols, scale))
                elif isinstance(delta, (ChannelScale, LowRank)):
                    a, b = (torch.as_tensor(f, dtype=torch.float64) for f in delta.lora(base_bits[tensor]))
                    if storage_dtype is not None:
                        a, b = a.to(storage_dtype), b.to(storage_dtype)
                    self.factors.setdefault(path, []).append((vid, a.to(factor_dtype), b.to(factor_dtype)))
                else:
                    raise TypeError(f"{tensor}: {type(delta).__name__} has no unmerged runtime form")

    def _rows(self, vid, device):
        return torch.nonzero(self.row_versions.to(device) == vid).flatten()

    def _mask_hook(self, path):
        entries = self.scales[path]

        def hook(module, args):
            x = args[0]
            out = None
            for vid, cols, scale in entries:
                rows = self._rows(vid, x.device)
                if rows.numel() == 0:
                    continue
                if out is None:
                    out = x.clone()
                sub = out.index_select(0, rows)
                # Multiply in at least fp32, then round once to the compute dtype.
                work = torch.promote_types(x.dtype, torch.float32)
                cols_here = cols.to(x.device)
                sub[..., cols_here] = (sub[..., cols_here].to(work) * scale.to(x.device, work)).to(x.dtype)
                out.index_copy_(0, rows, sub)
            return None if out is None else (out, *args[1:])
        return hook

    def _lora_hook(self, path):
        entries = self.factors[path]

        def hook(module, args, output):
            x = args[0]
            y = None
            for vid, a, b in entries:
                rows = self._rows(vid, x.device)
                if rows.numel() == 0:
                    continue
                if y is None:
                    y = output.clone()
                xs = x.index_select(0, rows).to(a.dtype)
                delta = (xs @ a.to(x.device).T) @ b.to(x.device).T
                ys = y.index_select(0, rows).to(a.dtype) + delta
                y.index_copy_(0, rows, ys.to(output.dtype))
            return y
        return hook

    @contextmanager
    def active(self, row_versions):
        """Install the hooks for one forward, ``row_versions`` lists a version name per row."""
        self.row_versions = torch.tensor([self.ids[v] for v in row_versions])
        modules = dict(self.model.named_modules())
        try:
            for path in self.scales:
                self.handles.append(modules[path].register_forward_pre_hook(self._mask_hook(path)))
            for path in self.factors:
                self.handles.append(modules[path].register_forward_hook(self._lora_hook(path)))
            yield self
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            self.row_versions = None


@contextmanager
def materialized(model, edited_bits):
    """Temporarily replace weights with exact edited BF16 tensors, then restore the originals."""
    modules = dict(model.named_modules())
    saved = {}
    try:
        for tensor, bits in edited_bits.items():
            weight = modules[module_path(tensor, model)].weight
            saved[tensor] = weight.detach().clone()
            with torch.no_grad():
                weight.copy_(bits_to_tensor(bits).to(weight.dtype))
        yield model
    finally:
        with torch.no_grad():
            for tensor, original in saved.items():
                modules[module_path(tensor, model)].weight.copy_(original)


class BatchedVersions:
    """Vectorized per-row deltas: one gather per batch, no per-version Python loop or sync.

    Per edited module: an input-scale table [V, in] (ones where untouched) and low-rank
    tables A [V, r, in], B [V, out, r] (zeros where untouched). ``active(row_versions)``
    gathers each module's per-row tensors once, the hooks then apply
    x * S[row] before the base matmul and y += (x A[row]^T) B[row]^T after it.
    """

    def __init__(self, model, versions, base_bits, *, device=None):
        self.model = model
        self.names = ["base", *versions]
        self.ids = {name: i for i, name in enumerate(self.names)}
        modules = dict(model.named_modules())
        plan = {}
        for version, deltas in versions.items():
            for tensor, delta in deltas.items():
                plan.setdefault(module_path(tensor, model), []).append((self.ids[version], tensor, delta))
        self.tables, self.handles, self.rows = {}, [], {}
        for path, entries in plan.items():
            weight = modules[path].weight
            out, inputs = weight.shape
            dev = device or weight.device
            scale = torch.ones(len(self.names), inputs)
            rank = max([d.rank for _, _, d in entries if isinstance(d, LowRank)] or [0])
            a = torch.zeros(len(self.names), rank, inputs)
            b = torch.zeros(len(self.names), out, rank)
            for vid, tensor, delta in entries:
                if isinstance(delta, ChannelScale):
                    scale[vid, torch.as_tensor(delta.columns)] = torch.as_tensor(delta.scales)
                elif isinstance(delta, LowRank):
                    fa, fb = delta.lora()
                    a[vid, :delta.rank] = torch.as_tensor(fa, dtype=torch.float32)
                    b[vid, :, :delta.rank] = torch.as_tensor(fb, dtype=torch.float32)
                else:
                    raise TypeError(type(delta))
            has_scale = bool((scale != 1).any())
            self.tables[path] = (scale.to(dev) if has_scale else None, a.to(dev) if rank else None,
                                 b.to(dev) if rank else None)

    @contextmanager
    def active(self, row_versions):
        modules = dict(self.model.named_modules())
        index = None
        try:
            for path, (scale, a, b) in self.tables.items():
                if index is None:
                    device = (scale if scale is not None else a).device
                    index = torch.tensor([self.ids[v] for v in row_versions], device=device)
                if scale is not None:
                    rows_scale = scale.index_select(0, index)[:, None, :]

                    def pre(module, args, s=rows_scale):
                        x = args[0]
                        return ((x.float() * s).to(x.dtype), *args[1:])
                    self.handles.append(modules[path].register_forward_pre_hook(pre))
                if a is not None:
                    rows_a, rows_b = a.index_select(0, index), b.index_select(0, index)

                    def post(module, args, output, ra=rows_a, rb=rows_b):
                        x = args[0].float()
                        low = torch.einsum("bti,bri->btr", x, ra)
                        return (output.float() + torch.einsum("btr,bor->bto", low, rb)).to(output.dtype)
                    self.handles.append(modules[path].register_forward_hook(post))
            yield self
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
