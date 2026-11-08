"""Typed representations of an edit W -> W' of one output projection.

Shapes follow torch.nn.Linear: W is [out, in] and y = x W^T, so columns are input
channels. ``materialize`` reproduces the evaluated BF16 tensor bit for bit (callers
check it against the published SHA-256). ``lora`` gives factors (A [r, in], B [out, r])
with W' - W = B A in real arithmetic, serving them unmerged is not bit-identical to the
rounded checkpoint, which is why the model-level check exists.
"""
from dataclasses import dataclass

import numpy as np

from . import bf16

INDEX_BYTES, VALUE_BYTES, FACTOR_BYTES, SCALE_BYTES = 4, 2, 4, 4


@dataclass
class ChannelScale:
    """W'[:, j] = bf16(fp32(c_j) * W[:, j]), c_j = 0 written by assignment gives +0.0."""
    columns: np.ndarray
    scales: np.ndarray
    assigned_zero: np.ndarray
    kind = "channel_scale"

    def materialize(self, base_bits):
        out = np.array(base_bits, copy=True)
        values = bf16.from_f32(bf16.to_f32(base_bits[:, self.columns]) * self.scales[None, :].astype(np.float32))
        values[:, self.assigned_zero] = 0
        out[:, self.columns] = values
        return out

    def lora(self, base_bits):
        rank, inputs = len(self.columns), base_bits.shape[1]
        a = np.zeros((rank, inputs))
        a[np.arange(rank), self.columns] = 1.0
        b = bf16.to_f64(base_bits[:, self.columns]) * (self.scales.astype(np.float64) - 1.0)[None, :]
        return a, b

    @property
    def rank(self):
        return len(self.columns)

    def nbytes(self):
        return len(self.columns) * (INDEX_BYTES + SCALE_BYTES)

    def describe(self):
        values, counts = np.unique(self.scales, return_counts=True)
        return {"kind": self.kind, "columns": int(len(self.columns)),
                "scales": {str(np.float32(v)): int(c) for v, c in zip(values, counts)},
                "assigned_zero_columns": int(self.assigned_zero.sum()), "bytes": self.nbytes()}


@dataclass
class LowRank:
    """W' = bf16(fp32(W + B A)) on the columns where A is nonzero, then sparse fixups."""
    b: np.ndarray
    a: np.ndarray
    fixup_index: np.ndarray
    fixup_bits: np.ndarray
    kind = "low_rank"

    @property
    def support(self):
        return np.flatnonzero(np.any(self.a != 0, axis=0))

    @property
    def rank(self):
        return self.a.shape[0]

    def materialize(self, base_bits, *, fixups=True):
        out = np.array(base_bits, copy=True)
        cols = self.support
        out[:, cols] = bf16.from_f64(bf16.to_f64(base_bits[:, cols]) + self.b @ self.a[:, cols])
        if fixups and len(self.fixup_index):
            out.reshape(-1)[self.fixup_index] = self.fixup_bits
        return out

    def lora(self, base_bits=None):
        return self.a, self.b

    def nbytes(self):
        return (self.rank * (self.b.shape[0] + len(self.support)) * FACTOR_BYTES + len(self.support) * INDEX_BYTES
                + len(self.fixup_index) * (INDEX_BYTES + VALUE_BYTES))

    def describe(self):
        return {"kind": self.kind, "rank": int(self.rank), "support_columns": int(len(self.support)),
                "fixups": int(len(self.fixup_index)), "bytes": self.nbytes()}


@dataclass
class SparseOverlay:
    """Changed BF16 entries stored as (flat index, new bits): the authors' small-edit format."""
    index: np.ndarray
    bits: np.ndarray
    kind = "sparse_overlay"

    def materialize(self, base_bits):
        out = np.array(base_bits, copy=True)
        out.reshape(-1)[self.index] = self.bits
        return out

    def nbytes(self):
        return len(self.index) * (INDEX_BYTES + VALUE_BYTES)

    def describe(self):
        return {"kind": self.kind, "entries": int(len(self.index)), "bytes": self.nbytes()}


def serving_cost(delta, shape):
    """Extra multiply-adds per token for serving the edit unmerged next to a shared base."""
    out, inputs = shape
    if isinstance(delta, ChannelScale):
        return {"input_mask_madds": int(delta.rank), "lora_madds": int(delta.rank * (out + 1))}
    if isinstance(delta, LowRank):
        return {"lora_madds": int(delta.rank * (out + len(delta.support))),
                "fixup_madds": int(len(delta.fixup_index))}
    if isinstance(delta, SparseOverlay):
        return {"sparse_madds": int(len(delta.index))}
    raise TypeError(type(delta))
