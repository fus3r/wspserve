"""A resident multi-version engine for weight-space programming sessions.

One copy of the base model stays loaded. Every edit version is a set of typed deltas
(input-channel scales, low-rank terms, or column overlays for anything else), compiled
bit-exactly from the BF16 tensors an editor writes. Each row of a batch carries its own
version, so the original and edited halves of a paired pass, and requests from several
sessions, run in one batch with no weight swap and no engine rebuild.

Generation follows the Sparks of WSP protocol (temperature, top-p, top-k, presence
penalty, a sampling seed fixed per problem and sample index, shared by both models). Rows
leave the batch as soon as they finish, reach their cap, or are caught looping, and the
batch is compacted. Between decode chunks the caller may run short requests (probes).
"""
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

from wspdelta import bf16
from wspdelta.delta import ChannelScale, LowRank, SparseOverlay
from wspdelta.runtime import module_path
from wspdelta.structure import analyze

REPO, REVISION = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"


def load_model(st, device):
    from transformers import Qwen3_5ForCausalLM

    model, info = Qwen3_5ForCausalLM.from_pretrained(REPO, revision=REVISION, dtype=torch.bfloat16,
                                                     output_loading_info=True)
    assert not any(info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys")), info
    # from_pretrained(dtype=bf16) also casts the checkpoint's FP32 tensors (A_log), restore them.
    restored = []
    params = dict(model.named_parameters())
    for name in st.names():
        dtype, _ = st.info(name)
        if dtype == "F32" and name.startswith("model.language_model."):
            target = params[name.replace("model.language_model.", "model.")]
            target.data = torch.from_numpy(st.raw(name).astype(np.float32))
            restored.append(name)
    return model.to(device).eval(), restored


@dataclass
class Sampling:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5


@dataclass
class Row:
    version: str
    prompt: list
    seed: int
    max_new: int
    sampling: Sampling = field(default_factory=Sampling)
    loop_stop: bool = True
    tokens: list = field(default_factory=list)
    finish: str = ""
    penalize: list = field(default_factory=list)   # earlier tokens of the same sample (presence penalty)


class LoopDetector:
    """Causal repeated-n-gram detector (n=32, window 1,024, threshold 0.75, every 128 tokens).

    """

    def __init__(self, n=32, window=1024, threshold=0.75, step=128):
        self.n, self.window, self.threshold, self.step = n, window, threshold, step
        rng = np.random.default_rng(0)
        self.weights = rng.integers(1, 2**63, size=n, dtype=np.uint64) | np.uint64(1)

    def looping(self, tokens):
        if len(tokens) < self.window or len(tokens) % self.step:
            return False
        tail = np.asarray(tokens[-self.window:], dtype=np.uint64)
        view = np.lib.stride_tricks.sliding_window_view(tail, self.n)
        with np.errstate(over="ignore"):
            grams = (view * self.weights).sum(axis=1, dtype=np.uint64)
        return 1.0 - len(np.unique(grams)) / len(grams) >= self.threshold


class Engine:
    def __init__(self, device=None, *, model=None, tokenizer=None, base_source=None):
        """Optionally use a resident model, tokenizer and source exposing ``bf16(name)``."""
        if model is None:
            from huggingface_hub import hf_hub_download
            from transformers import AutoTokenizer

            from wspdelta.safetensors_raw import SafeTensors

            self.device = device or ("cuda" if torch.cuda.is_available()
                                     else "mps" if torch.backends.mps.is_available() else "cpu")
            self.st = SafeTensors(hf_hub_download(REPO, "model.safetensors-00001-of-00001.safetensors", revision=REVISION))
            self.tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
            self.model, _ = load_model(self.st, self.device)
        else:
            if tokenizer is None or base_source is None:
                raise ValueError("model injection requires tokenizer and base_source")
            self.device = device or str(next(model.parameters()).device)
            self.model = model.to(self.device).eval()
            self.tok, self.st = tokenizer, base_source
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.pad = self.tok.pad_token_id
        self.versions = {"base": {}}          # version id -> {tensor name: typed delta}
        self._bits = {}
        self._tables = None
        self.detector = LoopDetector()
        self.stats = {"forward_rows": 0, "decode_steps": 0, "generated_tokens": 0, "loop_stops": 0,
                      "compactions": 0}

    # ------------------------------------------------------------------ versions

    def base_bits(self, name):
        if name not in self._bits:
            self._bits[name] = self.st.bf16(name)
        return self._bits[name]

    def register(self, edited_bits):
        """Register a version from edited BF16 tensors, returns its content id."""
        digest = hashlib.sha256()
        for name in sorted(edited_bits):
            digest.update(name.encode())
            digest.update(bf16.sha256(edited_bits[name]).encode())
        vid = "base" if not edited_bits else digest.hexdigest()[:12]
        if vid not in self.versions:
            deltas = {}
            for name, bits in edited_bits.items():
                best, options, _ = analyze(self.base_bits(name), bits)
                deltas[name] = best
            self.versions[vid] = deltas
            self._tables = None
        return vid

    def register_float32(self, tensors):
        """What an editor's save_edit writes: float32 tensors, rounded to BF16 as the evaluation does."""
        bits = {}
        for name, value in tensors.items():
            rounded = bf16.from_f32(np.asarray(value, dtype=np.float32))
            if not np.array_equal(rounded, self.base_bits(name)):
                bits[name] = rounded
        return self.register(bits)

    def materialized(self, vid):
        return {name: d.materialize(self.base_bits(name)) for name, d in self.versions[vid].items()}

    def _build_tables(self):
        """Per edited module, one table row per registered version."""
        names = list(self.versions)
        index = {v: i for i, v in enumerate(names)}
        plan = {}
        for vid, deltas in self.versions.items():
            for tensor, delta in deltas.items():
                plan.setdefault(module_path(tensor, self.model), []).append((index[vid], tensor, delta))
        modules = dict(self.model.named_modules())
        tables = {}
        for path, entries in plan.items():
            out, inputs = modules[path].weight.shape
            V = len(names)
            scale = torch.ones(V, inputs)
            rank = max([d.rank for _, _, d in entries if isinstance(d, LowRank)] or [0])
            a, b = torch.zeros(V, rank, inputs), torch.zeros(V, out, rank)
            overlay_cols = sorted({int(c) for _, t, d in entries if isinstance(d, SparseOverlay)
                                   for c in np.unique(d.index % inputs)})
            col_index = {c: i for i, c in enumerate(overlay_cols)}
            over = torch.zeros(V, out, len(overlay_cols))
            for vi, tensor, delta in entries:
                if isinstance(delta, ChannelScale):
                    scale[vi, torch.as_tensor(delta.columns)] = torch.as_tensor(delta.scales)
                elif isinstance(delta, LowRank):
                    fa, fb = delta.lora()
                    a[vi, :delta.rank] = torch.as_tensor(fa, dtype=torch.float32)
                    b[vi, :, :delta.rank] = torch.as_tensor(fb, dtype=torch.float32)
                else:  # exact column overlay: y += x[J] @ (W' - W)[:, J]^T
                    base = self.base_bits(tensor)
                    edited = delta.materialize(base)
                    cols = np.unique(delta.index % inputs)
                    diff = bf16.to_f64(edited[:, cols]) - bf16.to_f64(base[:, cols])
                    for j, c in enumerate(cols):
                        over[vi, :, col_index[int(c)]] = torch.as_tensor(diff[:, j], dtype=torch.float32)
            dev = self.device
            tables[path] = {
                "scale": scale.to(dev) if bool((scale != 1).any()) else None,
                "a": a.to(dev) if rank else None, "b": b.to(dev) if rank else None,
                "cols": torch.as_tensor(overlay_cols, dtype=torch.long, device=dev) if overlay_cols else None,
                "over": over.to(dev) if overlay_cols else None,
            }
        self._tables = (index, tables)

    @contextmanager
    def rows(self, row_versions):
        """Install per-row deltas for one batch whose rows carry these versions."""
        if self._tables is None:
            self._build_tables()
        index, tables = self._tables
        handles = []
        modules = dict(self.model.named_modules())
        rows = torch.tensor([index[v] for v in row_versions], device=self.device)
        try:
            for path, t in tables.items():
                if t["scale"] is not None:
                    s = t["scale"].index_select(0, rows)[:, None, :]

                    def pre(module, args, s=s):
                        x = args[0]
                        return ((x.float() * s).to(x.dtype), *args[1:])
                    handles.append(modules[path].register_forward_pre_hook(pre))
                if t["a"] is not None or t["over"] is not None:
                    ra = t["a"].index_select(0, rows) if t["a"] is not None else None
                    rb = t["b"].index_select(0, rows) if t["b"] is not None else None
                    ro = t["over"].index_select(0, rows) if t["over"] is not None else None
                    cols = t["cols"]

                    def post(module, args, output, ra=ra, rb=rb, ro=ro, cols=cols):
                        x = args[0].float()
                        y = output.float()
                        if ra is not None:
                            y = y + torch.einsum("btr,bor->bto", torch.einsum("bti,bri->btr", x, ra), rb)
                        if ro is not None:
                            y = y + torch.einsum("btj,boj->bto", x.index_select(2, cols), ro)
                        return y.to(output.dtype)
                    handles.append(modules[path].register_forward_hook(post))
            yield
        finally:
            for h in handles:
                h.remove()

    # ------------------------------------------------------------------ teacher forcing

    @torch.no_grad()
    def teacher_forced(self, items, *, hidden_layers=(), mlp_layers=(), topk=0, token_budget=8192, slice_len=512):
        """items: list of (version, token ids). Returns per-item token logprobs and requested states."""
        results = [None] * len(items)
        order = sorted(range(len(items)), key=lambda i: -len(items[i][1]))
        start = 0
        layers = self.model.model.layers
        while start < len(order):
            width = len(items[order[start]][1])
            take = max(1, token_budget // max(width, 1))
            chunk = order[start:start + take]
            start += take
            ids = torch.full((len(chunk), width), self.pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for r, i in enumerate(chunk):
                seq = items[i][1]
                ids[r, :len(seq)] = torch.tensor(seq)
                mask[r, :len(seq)] = 1
            captured, handles = {}, []
            for l in mlp_layers:
                def grab(module, args, l=l):
                    captured[l] = args[0].detach().float().cpu()
                handles.append(layers[l].mlp.down_proj.register_forward_pre_hook(grab))
            try:
                with self.rows([items[i][0] for i in chunk]):
                    out = self.model.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                           use_cache=False, output_hidden_states=bool(hidden_layers))
            finally:
                for h in handles:
                    h.remove()
            head = self.model.lm_head.weight
            for r, i in enumerate(chunk):
                n = len(items[i][1])
                target = torch.tensor(items[i][1][1:], device=self.device)
                token_lp, top_v, top_i = [], [], []
                for s0 in range(0, n, slice_len):   # never materialize [T, vocab] for the whole batch
                    lp = torch.log_softmax((out.last_hidden_state[r, s0:min(n, s0 + slice_len)] @ head.T).float(), -1)
                    t = target[s0:min(n - 1, s0 + slice_len)]
                    token_lp.append(lp[:len(t)].gather(1, t[:, None])[:, 0])
                    if topk:
                        v, k = lp.topk(topk, dim=-1)
                        top_v.append(v)
                        top_i.append(k)
                entry = {"token_logprobs": torch.cat(token_lp).cpu().numpy()}
                if topk:
                    entry["topk_logprobs"] = torch.cat(top_v).cpu().numpy()
                    entry["topk_ids"] = torch.cat(top_i).cpu().numpy()
                for l in hidden_layers:
                    entry[f"hidden_{l}"] = out.hidden_states[l][r, :n].float().cpu().numpy()
                for l in mlp_layers:
                    entry[f"mlp_act_{l}"] = captured[l][r, :n].numpy()
                results[i] = entry
            self.stats["forward_rows"] += len(chunk)
        return results

    # ------------------------------------------------------------------ generation

    def _sample(self, logits, rows, seen, generators):
        """vLLM-style: presence penalty on generated tokens, temperature, top-k, top-p, one uniform per row."""
        logits = logits.float()
        penalties = torch.tensor([r.sampling.presence_penalty for r in rows], device=logits.device)
        logits = logits - penalties[:, None] * seen
        greedy = [r.sampling.temperature <= 0 for r in rows]
        k = max(r.sampling.top_k for r in rows)
        values, ids = logits.topk(k, dim=-1)
        out = ids[:, 0].clone()
        temps = torch.tensor([max(r.sampling.temperature, 1e-6) for r in rows], device=logits.device)
        probs = torch.softmax(values / temps[:, None], dim=-1)
        ks = torch.tensor([r.sampling.top_k for r in rows], device=logits.device)
        probs = probs * (torch.arange(k, device=logits.device)[None, :] < ks[:, None])
        probs = probs / probs.sum(-1, keepdim=True)
        cumulative = probs.cumsum(-1)
        top_p = torch.tensor([r.sampling.top_p for r in rows], device=logits.device)
        keep = (cumulative - probs) < top_p[:, None]
        probs = probs * keep
        probs = probs / probs.sum(-1, keepdim=True)
        u = torch.tensor([float(torch.rand(1, generator=g)) for g in generators], device=logits.device)
        choice = (probs.cumsum(-1) < u[:, None]).sum(-1).clamp(max=k - 1)
        sampled = ids.gather(1, choice[:, None])[:, 0]
        greedy_mask = torch.tensor(greedy, device=logits.device)
        return torch.where(greedy_mask, out, sampled)

    def _generation_inputs(self, rows):
        width = max(len(r.prompt) for r in rows)
        ids = torch.full((len(rows), width), self.pad, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, width - len(r.prompt):] = torch.tensor(r.prompt)
            mask[i, width - len(r.prompt):] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        pos = (mask.cumsum(-1) - 1).clamp_min(0)
        seen = torch.zeros((len(rows), self.model.config.vocab_size), device=self.device)
        for i, r in enumerate(rows):
            if r.penalize:
                seen[i, torch.tensor(sorted(set(r.penalize)), device=self.device)] = 1.0
        return ids, mask, pos, seen

    def _chunked_prefill(self, rows, step_input, mask, step_pos, prefill_chunk):
        """Feed all but the last prefill_chunk prompt columns through the cache in chunks, so a
        long prompt never builds a full T x T attention score matrix. Returns the cache and the
        remaining (last) chunk's inputs, the 2-D mask keeps covering every position."""
        width = step_input.shape[1]
        if not prefill_chunk or width <= prefill_chunk:
            return None, step_input, step_pos
        cache, start = None, 0
        last = width - prefill_chunk
        with self.rows([r.version for r in rows]):
            while start < last:
                end = min(start + prefill_chunk, last)
                out = self.model(input_ids=step_input[:, start:end], attention_mask=mask[:, :end],
                                 position_ids=step_pos[:, start:end], past_key_values=cache,
                                 use_cache=True, logits_to_keep=1)
                cache, start = out.past_key_values, end
        return cache, step_input[:, last:], step_pos[:, last:]

    @staticmethod
    def _merge_caches(cache, incoming):
        """Exact merge for Qwen3.5's dynamic full-attention and Gated DeltaNet layers."""
        from transformers.cache_utils import DynamicLayer, LinearAttentionLayer

        if len(cache.layers) != len(incoming.layers):
            raise NotImplementedError("cannot exactly merge caches with different layer counts")
        for left, right in zip(cache.layers, incoming.layers):
            if type(left) is not type(right) or type(left) not in (DynamicLayer, LinearAttentionLayer):
                raise NotImplementedError(f"no exact cache merge for {type(left).__name__} / {type(right).__name__}")
        for left, right in zip(cache.layers, incoming.layers):
            if type(left) is DynamicLayer:
                width = max(left.keys.shape[-2], right.keys.shape[-2])
                for name in ("keys", "values"):
                    a, b = getattr(left, name), getattr(right, name)
                    a = torch.nn.functional.pad(a, (0, 0, width - a.shape[-2], 0))
                    b = torch.nn.functional.pad(b, (0, 0, width - b.shape[-2], 0))
                    setattr(left, name, torch.cat((a, b), dim=0))
            else:
                for name in ("conv_states", "recurrent_states"):
                    a, b = getattr(left, name), getattr(right, name)
                    for i in range(left.number_of_states):
                        a[i] = torch.cat((a[i], b[i]), dim=0)
        return cache

    def _trim_padding(self, cache, mask):
        """Drop leading cache columns that are padding for every remaining row.

        Rows that join a running batch are left-padded to its length, without trimming, a batch
        that never drains keeps every row attending over an ever longer, mostly masked history.
        Those columns are masked for all rows, so removing them is exact."""
        lead = int((mask.cumsum(1) == 0).sum(1).min())
        if lead == 0:
            return cache, mask
        from transformers.cache_utils import DynamicLayer

        for layer in cache.layers:
            if type(layer) is DynamicLayer:
                layer.keys = layer.keys[..., lead:, :].contiguous()
                layer.values = layer.values[..., lead:, :].contiguous()
        self.stats["trimmed_columns"] = self.stats.get("trimmed_columns", 0) + lead
        return cache, mask[:, lead:]

    @torch.no_grad()
    def generate(self, rows, *, chunk=32, between_chunks=None, admit=None, on_finish=None, max_rows=None,
                 prefill_chunk=None):
        """Decode rows, optionally admitting new rows after each compacted chunk.

        ``admit()`` retains ownership of rows beyond the available ``max_rows`` slots,
        they are left untouched. The returned sequence also includes accepted new rows.
        ``on_finish(row)`` runs immediately when a row acquires its finish reason.
        """
        if max_rows is not None and max_rows < 1:
            raise ValueError("max_rows must be positive")
        if not rows:
            return rows
        result = rows if admit is None else list(rows)
        active = list(rows[:max_rows])
        generators = [torch.Generator().manual_seed(int(r.seed)) for r in active]
        step_input, mask, step_pos, seen = self._generation_inputs(active)
        cache, step_input, step_pos = self._chunked_prefill(active, step_input, mask, step_pos, prefill_chunk)

        def record_tokens(batch, nxt):
            for r, token in zip(batch, nxt.tolist()):
                if r.finish:
                    continue
                r.tokens.append(token)
                self.stats["generated_tokens"] += 1
                if token in self.eos:
                    r.finish = "stop"
                elif len(r.tokens) >= r.max_new:
                    r.finish = "length"
                elif r.loop_stop and self.detector.looping(r.tokens):
                    r.finish = "loop"
                    self.stats["loop_stops"] += 1
                if r.finish and on_finish is not None:
                    on_finish(r)

        while active:
            with self.rows([r.version for r in active]):
                for _ in range(chunk):
                    out = self.model(input_ids=step_input, attention_mask=mask, position_ids=step_pos,
                                     past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = out.past_key_values
                    nxt = self._sample(out.logits[:, -1], active, seen, generators)
                    self.stats["decode_steps"] += 1
                    record_tokens(active, nxt)
                    seen[torch.arange(len(active), device=self.device), nxt] = 1.0
                    step_input = nxt[:, None]
                    step_pos = step_pos[:, -1:] + 1
                    mask = torch.cat([mask, torch.ones((len(active), 1), dtype=mask.dtype, device=self.device)], 1)
                    if all(r.finish for r in active):
                        break
            keep = [slot for slot, r in enumerate(active) if not r.finish]
            if len(keep) < len(active):
                if keep:
                    index = torch.tensor(keep, device=self.device)
                    cache.reorder_cache(index)
                    mask, step_input, step_pos, seen = (t.index_select(0, index) for t in (mask, step_input, step_pos, seen))
                    generators = [generators[slot] for slot in keep]
                    self.stats["compactions"] += 1
                    cache, mask = self._trim_padding(cache, mask)
                active = [active[slot] for slot in keep]
            if between_chunks is not None and (active or admit is not None):
                between_chunks()
            if admit is None:
                continue
            incoming = list(admit())
            if max_rows is not None:
                incoming = incoming[:max(0, max_rows - len(active))]
            if not incoming:
                continue
            result.extend(incoming)
            new_generators = [torch.Generator().manual_seed(int(r.seed)) for r in incoming]
            ids, new_mask, pos, new_seen = self._generation_inputs(incoming)
            with self.rows([r.version for r in incoming]):
                out = self.model(input_ids=ids, attention_mask=new_mask, position_ids=pos,
                                 use_cache=True, logits_to_keep=1)
            new_cache = out.past_key_values
            nxt = self._sample(out.logits[:, -1], incoming, new_seen, new_generators)
            self.stats["decode_steps"] += 1
            record_tokens(incoming, nxt)
            new_seen[torch.arange(len(incoming), device=self.device), nxt] = 1.0
            new_input, new_pos = nxt[:, None], pos[:, -1:] + 1
            new_mask = torch.cat((new_mask, torch.ones_like(new_input)), dim=1)
            keep = [slot for slot, r in enumerate(incoming) if not r.finish]
            if not keep:
                continue
            if len(keep) < len(incoming):
                index = torch.tensor(keep, device=self.device)
                new_cache.reorder_cache(index)
                new_mask, new_input, new_pos, new_seen = (t.index_select(0, index) for t in
                                                         (new_mask, new_input, new_pos, new_seen))
                new_generators = [new_generators[slot] for slot in keep]
                self.stats["compactions"] += 1
            if active:
                cache = self._merge_caches(cache, new_cache)
                width = max(mask.shape[1], new_mask.shape[1])
                mask = torch.cat((torch.nn.functional.pad(mask, (width - mask.shape[1], 0)),
                                  torch.nn.functional.pad(new_mask, (width - new_mask.shape[1], 0))), dim=0)
                step_input, step_pos, seen = (torch.cat((a, b), dim=0) for a, b in
                                             ((step_input, new_input), (step_pos, new_pos), (seen, new_seen)))
                generators.extend(new_generators)
            else:
                cache, mask, step_input, step_pos, seen = new_cache, new_mask, new_input, new_pos, new_seen
                generators = new_generators
            active.extend(incoming[slot] for slot in keep)
        return result
