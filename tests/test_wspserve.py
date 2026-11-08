"""CPU-only semantic checks, no checkpoint downloads, HTTP requests or live session files."""
import hashlib
import json
import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from wspdelta import bf16
from wspdelta.delta import ChannelScale, LowRank
from wspdelta.runtime import materialized
from wspdelta.structure import analyze
from wspserve import server
from wspserve.engine import Engine, LoopDetector, Row, Sampling


class ToyTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [2 + ord(c) % 95 for c in text[:8]]}

    def apply_chat_template(self, messages, *, tokenize=True, **kwargs):
        text = " ".join(m["content"] for m in messages)
        return self(text) if tokenize else text

    def decode(self, tokens):
        return " ".join(map(str, tokens))


@pytest.fixture(scope="module", autouse=True)
def cpu_only():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with pytest.MonkeyPatch.context() as patch:
            def no_mps(*args, **kwargs):
                pytest.fail("CPU tests must not seed or select MPS")
            patch.setattr(torch.mps, "manual_seed", no_mps)
            patch.setattr(torch.backends.mps, "is_available", no_mps)
            yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def engine():
    config = Qwen3_5TextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        max_position_embeddings=512, pad_token_id=0, eos_token_id=None,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1]},
        attn_implementation="eager",
    )
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1234)
        model = Qwen3_5ForCausalLM(config).float().eval()
    # BF16 checkpoint values, with FP32 computation to avoid BF16 logit ties.
    tensors = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            tensors[name] = bf16.from_f32(param.numpy())
            param.copy_(torch.from_numpy(bf16.to_f32(tensors[name])))
    result = Engine(device="cpu", model=model, tokenizer=ToyTokenizer(),
                    base_source=SimpleNamespace(bf16=tensors.__getitem__))
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
    return result


@pytest.fixture
def versions(engine):
    down = "model.layers.0.mlp.down_proj.weight"
    proj = "model.layers.1.self_attn.o_proj.weight"
    scaled = bf16.to_f32(engine.base_bits(down)).copy()
    scaled[:, ::2] *= np.float32(16.0)
    base = bf16.to_f32(engine.base_bits(proj))
    rng = np.random.default_rng(42)
    u = rng.standard_normal(base.shape[0]).astype(np.float32)
    u /= np.linalg.norm(u)
    projected = base - np.float32(8.0) * np.outer(u, u @ base)
    edits = [{down: bf16.from_f32(scaled)}, {proj: bf16.from_f32(projected)}]
    for edited, kind in zip(edits, (ChannelScale, LowRank)):
        name, bits = next(iter(edited.items()))
        delta, _, _ = analyze(engine.base_bits(name), bits)
        assert isinstance(delta, kind)
        np.testing.assert_array_equal(delta.materialize(engine.base_bits(name)), bits)
    vids = [engine.register(bits) for bits in edits]
    assert isinstance(engine.versions[vids[0]][down], ChannelScale)
    assert isinstance(engine.versions[vids[1]][proj], LowRank)
    return ["base", *vids]


def row(version, prompt, max_new, *, seed=17, sampling=None):
    return Row(version, list(prompt), seed, max_new,
               sampling=sampling or Sampling(temperature=0, top_k=20, presence_penalty=0), loop_stop=False)


@torch.no_grad()
def reference_tokens(model, prompt, edits, max_new):
    """Independent greedy oracle: full prefixes, materialized weights, no Engine hooks/cache."""
    ids = torch.tensor([prompt], device="cpu")
    tokens = []
    with materialized(model, edits):
        for _ in range(max_new):
            logits = model(input_ids=ids, use_cache=False).logits[0, -1]
            top = logits.topk(2).values
            assert float(top[0] - top[1]) > 1e-5, "reference has an ambiguous logit tie"
            token = int(logits.argmax())
            tokens.append(token)
            ids = torch.cat((ids, torch.tensor([[token]], device="cpu")), dim=1)
    return tokens


class _StopThread(BaseException):
    pass


class IsolatedService(server.Service):
    """Run the real thread, with test-only shutdown and log-completion signals."""
    def __init__(self, **kwargs):
        self.stop_requested = threading.Event()
        self.log_changed = threading.Condition()
        self.written = 0
        super().__init__(**kwargs)

    def _serve(self):
        self.thread = threading.current_thread()
        try:
            super()._serve()
        except _StopThread:
            pass

    def _run_probes(self):
        if self.stop_requested.is_set():
            raise _StopThread
        super()._run_probes()

    def _dump_log(self):
        with self.log_changed:
            super()._dump_log()
            self.written = len(self.log)
            self.log_changed.notify_all()


@pytest.fixture
def service_factory(engine, tmp_path, monkeypatch):
    practice = tmp_path / "practice"
    practice.mkdir()
    problems = [{"id": f"p{i}", "problem": f"Question {i}?", "answer": str(i)} for i in range(2)]
    for name in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"):
        (practice / name).write_text("\n".join(map(json.dumps, problems)) if name.startswith("gsm") else "")
    monkeypatch.setattr(server, "PRACTICE", practice)
    monkeypatch.setattr(server, "SESSIONS", tmp_path / "sessions")
    services = []

    def make(**kwargs):
        options = dict(chunk=2, loop_stop=False, gather_s=0, engine=engine)
        options.update(kwargs)
        service = IsolatedService(**options)
        services.append(service)
        return service

    yield make
    for service in services:
        service.stop_requested.set()
        service.wakeup.set()
        service.thread.join(timeout=10)
        assert not service.thread.is_alive(), "test service thread did not shut down"


def submit(call, *args, **kwargs):
    """Daemon caller plus bounded Future.result: a broken thread cannot hang pytest."""
    future = Future()

    def run():
        try:
            future.set_result(call(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)

    threading.Thread(target=run, daemon=True).start()
    return future


def wait_log(service, kind, count):
    with service.log_changed:
        def entries():
            return [e for e in service.log[:service.written] if e["kind"] == kind]
        assert service.log_changed.wait_for(lambda: len(entries()) >= count, timeout=10)
        return entries()


def records(response, half="original"):
    assert response["ok"], response
    return [json.loads(line) for line in Path(response[half]["file"]).read_text().splitlines()]


def test_per_row_versions_match_materialized_greedy(engine, versions):
    prompt = [2, 9, 17, 25]
    specs = [(v, prompt) for v in versions] + [(versions[2], [3, 8]), (versions[1], [4, 6, 8])]
    rows = [row(v, p, 10) for v, p in specs]
    engine.generate(rows, chunk=3)
    for generated, (version, tokens) in zip(rows, specs):
        edits = engine.materialized(version)
        assert generated.tokens == reference_tokens(engine.model, tokens, edits, 10)
        assert generated.finish == "length"
    # This fixture must exercise observable edits, rather than only equal base outputs.
    assert rows[1].tokens != rows[0].tokens
    assert rows[2].tokens != rows[0].tokens
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in engine.model.modules())


def test_compaction_preserves_remaining_tokens(engine, versions):
    caps = [2, 5, 11, 7]
    prompts = [[2, 4], [3, 5, 7], [6, 8, 10, 12], [9]]
    names = [versions[1], "base", versions[2], versions[1]]
    sampling = Sampling(temperature=0.8, top_k=8, top_p=0.9, presence_penalty=0.75)
    compact = [row(v, p, n, seed=i, sampling=sampling) for i, (v, p, n) in enumerate(zip(names, prompts, caps))]
    full = [row(v, p, max(caps), seed=i, sampling=sampling) for i, (v, p) in enumerate(zip(names, prompts))]
    sizes = []
    handle = engine.model.register_forward_pre_hook(lambda m, a, kw: sizes.append(kw["input_ids"].shape[0]),
                                                     with_kwargs=True)
    try:
        engine.generate(compact, chunk=3)
    finally:
        handle.remove()
    assert sorted(set(sizes), reverse=True) == [4, 3, 2, 1]
    assert engine.stats["compactions"] == 3
    engine.generate(full, chunk=3)
    for short, long, cap in zip(compact, full, caps):
        assert short.finish == "length" and len(short.tokens) == cap
        assert short.tokens == long.tokens[:cap]


@pytest.mark.parametrize("boundary", [1, 2, 4])
@pytest.mark.parametrize("prompt_length", [2, 20])
def test_join_matches_materialized_greedy_and_preserves_running_rows(engine, versions, boundary, prompt_length):
    running = [row(v, [2, 9, 17, 25], n) for v, n in zip(versions, [3, 17, 21])]
    incoming = [row(v, list(range(2, 2 + prompt_length - i % 2)), 6 + i) for i, v in enumerate(versions)]
    expected = [reference_tokens(engine.model, r.prompt, engine.materialized(r.version), r.max_new)
                for r in running + incoming]
    calls, events = [], []

    def admit():
        calls.append([len(r.tokens) for r in running])
        return incoming if len(calls) == boundary else []

    result = engine.generate(running, chunk=2, admit=admit, on_finish=lambda r: events.append((id(r), r.finish)))
    assert calls[boundary - 1] == [min(2 * boundary, r.max_new) for r in running]
    assert result == running + incoming
    for generated, tokens in zip(result, expected):
        assert generated.tokens == tokens
        assert generated.finish == "length"
    assert sorted(events) == sorted((id(r), "length") for r in result)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in engine.model.modules())


def test_join_preserves_sampled_generators(engine, versions):
    sampling = Sampling(temperature=0.8, top_k=12, top_p=0.85, presence_penalty=1.25)
    running = [row(v, [2, 7, 11], 16, seed=30 + i, sampling=sampling) for i, v in enumerate(versions)]
    incoming = [row(v, [3, 8, 12, 19, 21, 23, 25, 27, 29] if i < 2 else [4, 9],
                    9, seed=40 + i, sampling=sampling)
                for i, v in enumerate(versions)]
    expected = []
    for r in running + incoming:
        alone = row(r.version, r.prompt, r.max_new, seed=r.seed, sampling=sampling)
        engine.generate([alone], chunk=3)
        expected.append(alone.tokens)
    calls = 0

    def admit():
        nonlocal calls
        calls += 1
        return incoming[:2] if calls == 1 else incoming[2:] if calls == 3 else []

    engine.generate(running, chunk=3, admit=admit)
    assert [r.tokens for r in running + incoming] == expected


def test_finish_callback_runs_before_chunk_boundary_and_on_prefill(engine):
    long, short, instant = row("base", [2, 4], 12), row("base", [3, 5], 2), row("base", [6, 8], 1)
    events, boundaries = [], []

    def admit():
        boundaries.append(len(long.tokens))
        return [instant] if len(boundaries) == 1 else []

    engine.generate([long, short], chunk=5, admit=admit,
                    on_finish=lambda r: events.append((id(r), len(long.tokens), len(boundaries))))
    assert events == [(id(short), 2, 0), (id(instant), 5, 1), (id(long), 12, 2)]


def test_engine_max_rows_leaves_excess_with_caller(engine):
    first, excess = row("base", [2, 4], 4), row("base", [3, 5], 3)
    waiting = [row("base", [6, 8], 3), row("base", [7, 9], 3)]
    sizes = []
    calls = 0

    def admit():
        nonlocal calls
        calls += 1
        return waiting if calls <= 2 else []

    handle = engine.model.register_forward_pre_hook(lambda m, a, kw: sizes.append(kw["input_ids"].shape[0]),
                                                     with_kwargs=True)
    try:
        # Returning too many rows cannot exceed the engine's cap or consume the caller's rows.
        returned = engine.generate([first, excess], chunk=2, max_rows=1, admit=admit)
    finally:
        handle.remove()
    assert max(sizes) == 1
    assert first.finish == waiting[0].finish == "length"
    assert not excess.tokens and not excess.finish
    assert not waiting[1].tokens and not waiting[1].finish
    assert waiting[0] in returned and waiting[1] not in returned


def test_cache_merge_rejects_unknown_layers():
    cache = SimpleNamespace(layers=[object()])
    with pytest.raises(NotImplementedError, match="no exact cache merge"):
        Engine._merge_caches(cache, cache)


def test_sampling_seeds_and_original_edited_pairing(engine, versions, service_factory, monkeypatch):
    sampling = Sampling(temperature=0.7, top_k=12, top_p=0.85, presence_penalty=1.25)
    rows = [row(v, [2, 7, 11], 12, seed=109, sampling=sampling)
            for v in ("base", versions[1], "base", versions[1])]
    engine.generate(rows, chunk=2)
    assert rows[0].tokens == rows[2].tokens
    assert rows[1].tokens == rows[3].tokens
    service = service_factory(memo=False)
    service.session("pairs").version = versions[1]
    captured = []
    generate = engine.generate

    def capture(batch, **kwargs):
        captured.extend((r.version, r.seed) for r in batch)
        return generate(batch, **kwargs)

    monkeypatch.setattr(engine, "generate", capture)
    request = {"type": "generate", "model": "both", "problems": ["p0", "p1"], "n": 2, "max_tokens": 4}
    response = submit(service.forward, "pairs", request).result(timeout=10)
    assert response["ok"], response
    expected = [int.from_bytes(hashlib.sha256(f"{pid}:{k}".encode()).digest()[:4], "little")
                for pid in request["problems"] for k in range(2)]
    assert len(set(expected)) == 4
    assert captured == [(v, seed) for v in ("base", versions[1]) for seed in expected]
    wait_log(service, "generate", 1)


def test_sampling_presence_top_k_top_p_and_temperature(engine, monkeypatch):
    logits = torch.tensor(np.array([[3., 2., 1., 0.], *[np.log([0.4, 0.3, 0.2, 0.1])] * 2,
                                    np.log([0.5, 0.25, 0.125, 0.125]), np.log([0.4, 0.3, 0.2, 0.1]),
                                    np.log([0.6, 0.3, 0.09, 0.01]), np.log([0.4, 0.3, 0.2, 0.1])]),
                          dtype=torch.float32, device="cpu")
    settings = [Sampling(0, 1, 4, 2), Sampling(1, 1, 2, 0), Sampling(1, 0.5, 4, 0),
                Sampling(1, 0.5, 4, 0), Sampling(1, 0.5, 2, 0), Sampling(0.5, 1, 4, 0), Sampling(1, 1, 4, 0)]
    rows = [row("base", [2], 1, sampling=s) for s in settings]
    seen = torch.zeros_like(logits)
    seen[0, 0] = 1
    uniforms = iter([0.9, 0.99, 0.99, 0.99, 0.99, 0.7, 0.99])
    generators = [torch.Generator(device="cpu").manual_seed(i) for i in range(len(rows))]
    used = []

    def rand(size, *, generator):
        assert size == 1
        used.append(generator)
        return torch.tensor([next(uniforms)], device="cpu")

    monkeypatch.setattr(torch, "rand", rand)
    # Penalized greedy, top-k, nucleus crossing, exact boundary, k before p,
    # temperature before normalization, and an unrestricted tail draw.
    assert engine._sample(logits, rows, seen, generators).tolist() == [1, 1, 1, 0, 0, 0, 3]
    assert used == generators  # exactly one uniform per row, including greedy rows


def test_loop_detector_flags_expected_position():
    detector = LoopDetector()
    periodic = [3, 5, 7, 11] * 288
    positions = [n for n in range(1, len(periodic) + 1) if detector.looping(periodic[:n])]
    assert positions == [1024, 1152]
    nonrepeating = list(range(2048))
    assert not any(detector.looping(nonrepeating[:n]) for n in range(1, len(nonrepeating) + 1))


def test_service_memo_and_dedupe_within_batch(engine, service_factory):
    service = service_factory()
    request = {"type": "generate", "model": "original", "problems": ["p0"], "n": 2, "max_tokens": 5}
    first = submit(service.forward, "memo", request).result(timeout=10)
    original = records(first)
    batch = wait_log(service, "generate", 1)[0]
    assert batch["rows"] == 2 and batch["memo_hits"] == 0
    generated = engine.stats["generated_tokens"]
    assert generated == 10 and len(service.memo) == 2
    second = submit(service.forward, "memo", request).result(timeout=10)
    assert records(second) == original
    batch = wait_log(service, "generate", 2)[1]
    assert batch["rows"] == 0 and batch["memo_hits"] == 2
    assert second["memo_hits"] == 2
    assert engine.stats["generated_tokens"] == generated
    # Cold memo keys: an unedited session's two halves share each base sample.
    paired = submit(service.forward, "memo", dict(request, model="both", problems=["p1"])).result(timeout=10)
    assert records(paired) == records(paired, "edited")
    batch = wait_log(service, "generate", 3)[2]
    assert batch["rows"] == 2 and batch["memo_hits"] == 2
    assert engine.stats["generated_tokens"] == generated + 10


def test_teacher_forced_probe_has_priority_during_generation(engine, service_factory, monkeypatch):
    service = service_factory(memo=False)
    decoding, queued, release = (threading.Event() for _ in range(3))
    generate, put = engine.generate, service.probes.put

    def probe_put(job):
        put(job)
        queued.set()

    def interleave(rows, **kwargs):
        callback = kwargs.pop("between_chunks")
        first = True

        def between():
            nonlocal first
            if first:
                decoding.set()
                assert queued.wait(10), "probe was not submitted during decoding"
            callback()  # the real Service._run_probes must complete the queued request
            if first:
                first = False
                assert release.wait(10), "test did not release generation"

        return generate(rows, between_chunks=between, **kwargs)

    monkeypatch.setattr(service.probes, "put", probe_put)
    monkeypatch.setattr(engine, "generate", interleave)
    generation = submit(service.forward, "long", {"type": "generate", "model": "original",
                                                "problems": ["p0"], "max_tokens": 128})
    try:
        assert decoding.wait(10)
        assert service._generating and not generation.done()
        probe = submit(service.forward, "probe", {"type": "teacher_forced", "model": "original",
                                                  "texts": ["abcdef"],
                                                  "return": {"hidden_states": [0], "mlp_act": [0], "topk": 3}})
        result = probe.result(timeout=10)
        assert result["ok"], result
        assert not generation.done()
        entry = wait_log(service, "probe", 1)[0]
        assert entry["during_generation"]
        assert result["lengths"] == [6]
        assert result["files"]["token_logprobs"] == [1, 5]
        for key, shape in result["files"].items():
            values = np.load(Path(result["dir"]) / f"{key}.npy")
            assert list(values.shape) == shape and np.isfinite(values).all()
    finally:
        queued.set()
        release.set()
    assert generation.result(timeout=15)["ok"]
    wait_log(service, "generate", 1)
    assert [entry["kind"] for entry in service.log] == ["probe", "generate"]


def test_gather_window_shares_batches_between_alternating_sessions(service_factory, monkeypatch):
    service = service_factory(memo=False, gather_s=1.0)
    for sid in ("alice", "bob"):
        result = submit(service.forward, sid, {"type": "teacher_forced", "texts": ["warmup"]}).result(timeout=10)
        assert result["ok"], result
    gathering = threading.Event()
    get = service.gens.get

    def observed_get(*args, **kwargs):
        if kwargs.get("timeout"):
            gathering.set()
        return get(*args, **kwargs)

    monkeypatch.setattr(service.gens, "get", observed_get)
    for count, (first, second) in enumerate((("alice", "bob"), ("bob", "alice")), start=1):
        gathering.clear()
        request = {"type": "generate", "model": "original", "problems": ["p0"], "max_tokens": 4}
        a = submit(service.forward, first, request)
        assert gathering.wait(10), "first request did not wait for the other active session"
        assert not a.done()
        b = submit(service.forward, second, request)
        assert a.result(timeout=10)["ok"] and b.result(timeout=10)["ok"]
        # Logs are now per job, the shared batch id preserves the batching assertion.
        batch = wait_log(service, "generate", 2 * count)[-2:]
        assert [entry["sessions"] for entry in batch] == [[first], [second]]
        assert batch[0]["batch"] == batch[1]["batch"]
        assert all(entry["rows"] == 1 and entry["versions"] == ["base"] and not entry["joined"] for entry in batch)


@pytest.mark.parametrize("join", [True, False])
def test_generation_arriving_during_decode_joins_within_one_chunk(engine, versions, service_factory, monkeypatch, join):
    service = service_factory(memo=False, join=join, chunk=3)
    service.session("short").version = versions[1]
    request = {"type": "generate", "model": "both", "problems": ["p1"], "max_tokens": 5}
    alone = submit(service.forward, "short", request).result(timeout=10)
    expected = {half: records(alone, half) for half in ("original", "edited")}
    wait_log(service, "generate", 1)

    decoding, queued, completed, release = (threading.Event() for _ in range(4))
    generate, put = engine.generate, service.gens.put
    boundaries, admitted = [], []
    probe = server.Job(service.session("probe"), {"type": "teacher_forced", "texts": ["warmup"]})

    def queued_put(job):
        put(job)
        if job.session.sid == "short":
            queued.set()

    def interleave(rows, **kwargs):
        callback, finish = kwargs.pop("between_chunks"), kwargs.pop("on_finish")
        admission = kwargs.pop("admit")
        long_rows = list(rows)

        def between():
            boundaries.append([len(r.tokens) for r in long_rows])
            if len(boundaries) == 1:
                decoding.set()
                assert queued.wait(10), "second generation was not submitted during decoding"
            callback()

        def admit():
            assert probe.done.is_set() and probe.result["ok"], "probes must run before generation admission"
            new = admission()
            if new:
                admitted.append([len(r.tokens) for r in long_rows])
            return new

        def on_finish(r):
            finish(r)
            if join and r.max_new == request["max_tokens"] and not completed.is_set():
                # Both halves finish on the same step, wait after the last half is summarized.
                if service.session("short").passes == 2:
                    completed.set()
                    assert release.wait(10), "test did not release long generation"

        return generate(rows, between_chunks=between, admit=admit if admission else None,
                        on_finish=on_finish, **kwargs)

    monkeypatch.setattr(service.gens, "put", queued_put)
    monkeypatch.setattr(engine, "generate", interleave)
    long = submit(service.forward, "long", {"type": "generate", "model": "original",
                                           "problems": ["p0"], "max_tokens": 36})
    try:
        assert decoding.wait(10)
        service.probes.put(probe)
        short = submit(service.forward, "short", request)
        if join:
            assert completed.wait(10)
            assert not long.done()
            assert admitted == [[service.chunk]]
        actual = short.result(timeout=10)
        assert {half: records(actual, half) for half in expected} == expected
    finally:
        queued.set()
        release.set()
    assert long.result(timeout=10)["ok"]
    assert probe.done.is_set() and probe.result["ok"]
    entries = wait_log(service, "generate", 3)[1:]
    by_session = {e["sessions"][0]: e for e in entries}
    assert by_session["short"]["joined"] is join
    assert (by_session["short"]["batch"] == by_session["long"]["batch"]) is join
    assert by_session["short"]["rows"] == 2 and by_session["short"]["memo_hits"] == 0
    if join:
        assert entries[0]["sessions"] == ["short"]
        assert by_session["short"]["t0"] >= by_session["long"]["t0"]
        assert by_session["short"]["t1"] <= by_session["long"]["t1"]
    else:
        assert entries[0]["sessions"] == ["long"]
        assert not admitted
    assert service.session("short").passes == 2 and service.session("long").passes == 1


def test_join_shares_pending_base_rows_and_uses_memo_during_decode(engine, service_factory, monkeypatch):
    service = service_factory(chunk=2, max_rows=1)
    request = {"type": "generate", "model": "original", "problems": ["p0"], "max_tokens": 8}
    # A different cap gives an independent memo key usable while the long key is pending.
    cached_request = dict(request, max_tokens=2)
    expected = records(submit(service.forward, "cached", cached_request).result(timeout=10))
    wait_log(service, "generate", 1)
    generated = engine.stats["generated_tokens"]
    decoding, release = threading.Event(), threading.Event()
    generate = engine.generate

    def interleave(rows, **kwargs):
        callback = kwargs.pop("between_chunks")

        def between():
            if not decoding.is_set():
                decoding.set()
                assert release.wait(10)
            callback()

        return generate(rows, between_chunks=between, **kwargs)

    monkeypatch.setattr(engine, "generate", interleave)
    long = submit(service.forward, "long", request)
    assert decoding.wait(10)
    # Queue real jobs directly so both are present at the next boundary, without a race.
    shared = server.Job(service.session("shared"), dict(request, model="both"))
    cached = server.Job(service.session("memo-hit"), cached_request)
    service.gens.put(shared)
    service.gens.put(cached)
    release.set()
    assert cached.done.wait(10) and shared.done.wait(10)
    original = records(long.result(timeout=10))
    assert records(shared.result) == records(shared.result, "edited") == original
    assert records(cached.result) == expected
    assert engine.stats["generated_tokens"] == generated + request["max_tokens"]
    entries = wait_log(service, "generate", 4)[1:]
    by_session = {e["sessions"][0]: e for e in entries}
    assert by_session["shared"]["rows"] == 0 and by_session["shared"]["memo_hits"] == 2
    assert shared.result["memo_hits"] == 2 and cached.result["memo_hits"] == 1
    assert by_session["memo-hit"]["rows"] == 0 and by_session["memo-hit"]["joined"]
    assert by_session["shared"]["joined"]
    assert len({e["batch"] for e in entries}) == 1


@pytest.mark.parametrize("join,max_tokens", [(True, 5), (False, 1)])
def test_service_max_rows_preserves_queued_work(engine, service_factory, monkeypatch, join, max_tokens):
    service = service_factory(memo=False, max_rows=2, join=join)
    sizes = []
    handle = engine.model.register_forward_pre_hook(lambda m, a, kw: sizes.append(kw["input_ids"].shape[0]),
                                                     with_kwargs=True)
    request = {"type": "generate", "model": "original", "problems": ["p0", "p1"],
               "n": 3, "max_tokens": max_tokens}
    try:
        response = submit(service.forward, "capped", request).result(timeout=10)
    finally:
        handle.remove()
    generated = records(response)
    assert len(generated) == 6 and max(sizes) == 2
    assert {(r["id"], r["sample"]) for r in generated} == {(pid, k) for pid in request["problems"] for k in range(3)}
    assert all(r["tokens"] == max_tokens and r["finish"] == "length" for r in generated)
    entry = wait_log(service, "generate", 1)[0]
    assert entry["rows"] == 6 and entry["tokens"] == 6 * max_tokens
    assert service.session("capped").passes == 1


def test_cli_join_flags(monkeypatch):
    options = []

    def service(**kwargs):
        options.append(kwargs)
        return object()

    listener = SimpleNamespace(serve_forever=lambda: None)
    monkeypatch.setattr(server, "Service", service)
    monkeypatch.setattr(server, "ThreadingHTTPServer", lambda *args: listener)
    monkeypatch.setattr("sys.argv", ["wspserve"])
    server.main()
    assert options[-1]["join"] and options[-1]["max_rows"] == 128
    monkeypatch.setattr("sys.argv", ["wspserve", "--no-join", "--max-rows", "7"])
    server.main()
    assert not options[-1]["join"] and options[-1]["max_rows"] == 7


def test_register_float32_ignores_unchanged_and_is_content_addressed(engine):
    down = "model.layers.0.mlp.down_proj.weight"
    proj = "model.layers.1.self_attn.o_proj.weight"
    base = bf16.to_f32(engine.base_bits(down))
    other = bf16.to_f32(engine.base_bits(proj))
    assert engine.register_float32({down: base.copy(), proj: other.copy()}) == "base"
    # Values that change in FP32 but round back to the checkpoint are unchanged too.
    assert engine.register_float32({down: base * np.float32(1 + 2**-12)}) == "base"
    edited = base.copy()
    edited[:, [2, 7]] *= np.float32(0.5)
    first = engine.register_float32({down: edited})
    second = engine.register_float32({proj: other.copy(), down: edited.astype(np.float64)})
    assert first == second != "base"
    assert set(engine.versions[first]) == {down}
    assert len(engine.versions) == 2
    np.testing.assert_array_equal(engine.materialized(first)[down], bf16.from_f32(edited))
    different = edited.copy()
    different[:, [2, 7]] *= np.float32(0.5)
    assert engine.register_float32({down: different}) not in ("base", first)


def test_never_draining_batch_trims_padding_and_stays_exact(engine, versions, monkeypatch):
    """A new row joins at every chunk while older rows are still running, so the batch never
    drains. Padding columns that every remaining row masks are trimmed after compaction: the
    cache stays bounded, and every row still decodes exactly as it does alone."""
    rows = [row(versions[i % 3], [2 + i, 9, 17, 25, 30 + i][: 3 + i % 3], 5, seed=60 + i) for i in range(12)]
    expected = [reference_tokens(engine.model, r.prompt, engine.materialized(r.version), r.max_new) for r in rows]
    widths, queue_ = [], list(rows[1:])
    original = engine._trim_padding

    def spy(cache, mask):
        cache, mask = original(cache, mask)
        widths.append(mask.shape[1])
        return cache, mask

    monkeypatch.setattr(engine, "_trim_padding", spy)
    before = engine.stats.get("trimmed_columns", 0)
    engine.generate(rows[:1], chunk=2, admit=lambda: [queue_.pop(0)] if queue_ else [])
    assert [r.tokens for r in rows] == expected
    assert engine.stats.get("trimmed_columns", 0) > before
    # without trimming, the width would grow by about one chunk per admission (12 admissions)
    assert max(widths) <= 5 + 5 + 2


def test_chunked_prefill_and_carried_presence_penalty_match_one_shot(engine, versions):
    """Chunked prefill of long, unequal prompts gives the same greedy tokens as one-shot prefill,
    and `penalize` makes earlier tokens of a sample count for the presence penalty."""
    prompts = [list(range(2, 2 + 37)), [5, 9, 13, 21, 7] * 5, list(range(40, 40 + 23))]
    one = [row(versions[i], p, 8, seed=80 + i) for i, p in enumerate(prompts)]
    chunked = [row(versions[i], p, 8, seed=80 + i) for i, p in enumerate(prompts)]
    engine.generate(one, chunk=3)
    engine.generate(chunked, chunk=3, prefill_chunk=6)
    assert [r.tokens for r in chunked] == [r.tokens for r in one]
    sampling = Sampling(temperature=0, top_k=20, presence_penalty=50.0)
    plain = row(versions[0], prompts[0], 1, seed=90, sampling=sampling)
    engine.generate([plain], chunk=1)
    carried = row(versions[0], prompts[0], 1, seed=90, sampling=sampling)
    carried.penalize = [plain.tokens[0]]
    engine.generate([carried], chunk=1)
    assert carried.tokens[0] != plain.tokens[0]
