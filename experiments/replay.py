"""Replay the start of a released WSP session through two execution designs, on this Mac.

The trace is the p50c editor's own forward requests, in order (problem ids, sample counts,
models, probe texts), scaled down to fit a laptop: at most P problems and T texts per
request, n = 1, max_tokens = M. The edited version is the released p50c final edit
throughout (the intermediate edits were not released).

- naive: what the authors' harness does per pass: rebuild the engine (here a model reload
  from disk, vLLM's rebuild costs minutes and failed 25 times in the released runs), run
  the original and the edited halves one after the other with materialized weights, no
  memoization, no loop stop,
- wspserve: one resident engine, both halves in one batch, base samples memoized, rows
  caught looping stopped early.

Wall-clock and token counts are measured, accuracies are reported for both so the two
designs can be checked to answer the same questions.
"""
import argparse
import os
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compile_released_edits import rebuild_from_bundle  # noqa: E402
from wspdelta import bf16  # noqa: E402

OUT = ROOT / "results/replay.json"


def trace(logs, run, count, problems, texts, max_tokens):
    rows = [json.loads(line) for line in (logs / f"runs/{run}/tool_calls.jsonl").read_text().splitlines()]
    out = []
    for r in rows:
        if r["kind"] != "forward" or not r["result"].get("ok"):
            continue
        req = dict(r["request"])
        if req["type"] == "generate":
            if not req.get("problems"):
                continue
            out.append({"type": "generate", "model": req.get("model", "both"), "problems": req["problems"][:problems],
                        "n": 1, "max_tokens": max_tokens})
        else:
            if req.get("chats"):
                out.append({"type": "teacher_forced", "model": req.get("model", "original"),
                            "chats": req["chats"][:texts], "enable_thinking": req.get("enable_thinking", True),
                            "return": {"token_logprobs": True}})
            else:
                out.append({"type": "teacher_forced", "model": req.get("model", "original"),
                            "texts": req["texts"][:texts], "return": {"token_logprobs": True}})
        if len(out) >= count:
            break
    return out


def p50c_float32(engine, logs):
    m = json.loads((logs / "runs/p50c/edit/manifest.json").read_text())
    out = {}
    with np.load(logs / "runs/p50c/edit/edit.npz") as z:
        for i, t in enumerate(m["tensors"]):
            out[t["name"]] = bf16.to_f32(rebuild_from_bundle(engine.base_bits(t["name"]), t, z, i))
    return out


def run_native(traces, logs, join=True, gather_s=3.0):
    """One session per trace, replayed concurrently against one service."""
    from wspserve.server import Service

    svc = Service(join=join, gather_s=gather_s)
    edits = p50c_float32(svc.engine, logs)
    timings, results = {}, {}

    def replay(sid, requests):
        s = svc.session(sid)
        s.edits = dict(edits)
        s.version = svc.engine.register_float32(s.edits)
        start = time.time()
        res = []
        for req in requests:
            t = time.time()
            r = svc.forward(sid, req)
            res.append({"type": req["type"], "s": round(time.time() - t, 2), "ok": r.get("ok"),
                        "accuracy": {h: r[h]["accuracy"] for h in ("original", "edited") if h in r},
                        "memo_hits": r.get("memo_hits"), "wait_s": r.get("wait_s")})
        timings[sid] = time.time() - start
        results[sid] = res

    threads = [threading.Thread(target=replay, args=(f"bench{k}", reqs)) for k, reqs in enumerate(traces)]
    started = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {"wall_s": time.time() - started, "per_session_s": timings, "passes": results,
            "engine": dict(svc.engine.stats), "log": svc.log}


def run_naive(requests, logs):
    import torch

    from wspserve.engine import load_model
    from wspserve.engine import Engine, Row
    from wspserve.server import SUFFIX, Service, grade
    from wspdelta.runtime import materialized

    import gc

    engine = Engine()
    problems = Service.__new__(Service)
    problems.problems = {}
    for name in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"):
        for line in (logs / "practice" / name).read_text().split("\n"):
            if line.strip():
                r = json.loads(line)
                problems.problems[r["id"]] = r
    edits = p50c_float32(engine, logs)
    vid = "p50c"   # not registered: the naive engine runs with no hooks, weights swapped in place
    bits = {name: bf16.from_f32(value) for name, value in edits.items()}
    tok = engine.tok
    started = time.time()
    res, rebuild_s = [], []
    import hashlib
    for req in requests:
        t = time.time()
        if req["type"] == "generate":
            r0 = time.time()
            engine.model = None
            gc.collect()
            if engine.device == "mps":
                torch.mps.empty_cache()
            engine.model, _ = load_model(engine.st, engine.device)   # fresh engine per pass
            if engine.device == "mps":
                torch.mps.synchronize()
            rebuild_s.append(time.time() - r0)
            halves = {"original": ["base"], "edited": [vid], "both": ["base", vid]}[req["model"]]
            acc = {}
            for half, version in zip(["original", "edited"] if len(halves) == 2 else [req["model"]], halves):
                rows = []
                for pid in req["problems"]:
                    prompt = tok.apply_chat_template([{"role": "user", "content": problems.problems[pid]["problem"] + SUFFIX}],
                                                     tokenize=True, add_generation_prompt=True, enable_thinking=True)["input_ids"]
                    seed = int.from_bytes(hashlib.sha256(f"{pid}:0".encode()).digest()[:4], "little")
                    rows.append(Row(version="base", prompt=prompt, seed=seed, max_new=req["max_tokens"], loop_stop=False))
                swap = bits if version != "base" else {}
                with materialized(engine.model, swap):
                    engine.generate(rows)
                acc[half] = float(np.mean([grade(tok.decode(r.tokens), problems.problems[pid]["answer"])[1]
                                           for r, pid in zip(rows, req["problems"])]))
            res.append({"type": "generate", "s": round(time.time() - t, 2), "accuracy": acc})
        else:
            if req.get("chats"):
                texts = [tok.apply_chat_template(c, tokenize=False, enable_thinking=bool(req.get("enable_thinking")))
                         for c in req["chats"]]
            else:
                texts = req["texts"]
            ids = [tok(x, add_special_tokens=False)["input_ids"] for x in texts]
            swap = bits if req["model"] == "edited" else {}
            with materialized(engine.model, swap):
                engine.teacher_forced([("base", i) for i in ids])
            res.append({"type": "teacher_forced", "s": round(time.time() - t, 2)})
    return {"wall_s": time.time() - started, "passes": res, "engine": dict(engine.stats),
            "model_reload_s": rebuild_s}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--run", default="p50c")
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--problems", type=int, default=8)
    parser.add_argument("--texts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--modes", nargs="*", default=["naive", "native", "naive2", "native2"])
    parser.add_argument("--second-run", default="p25c")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--no-join", action="store_true", help="newly queued requests wait for the running batch")
    parser.add_argument("--gather", type=float, default=3.0)
    args = parser.parse_args()
    from wspserve import server

    server.PRACTICE = Path(os.environ.get("WSP_PRACTICE", args.logs / "practice"))
    server.SESSIONS = Path(args.out).parent / "replay_sessions"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    requests = trace(args.logs, args.run, args.requests, args.problems, args.texts, args.max_tokens)
    report = {"trace": args.run, "requests": len(requests),
              "scaling": {"problems": args.problems, "texts": args.texts, "n": 1, "max_tokens": args.max_tokens},
              "kinds": [r["type"] for r in requests], "join": not args.no_join, "gather_s": args.gather}
    if "naive" in args.modes:
        report["naive"] = run_naive(requests, args.logs)
        print("naive", round(report["naive"]["wall_s"], 1), "s", report["naive"]["engine"], flush=True)
    if "native" in args.modes:
        report["wspserve"] = run_native([requests], args.logs, join=not args.no_join, gather_s=args.gather)
        print("wspserve", round(report["wspserve"]["wall_s"], 1), "s", report["wspserve"]["engine"], flush=True)
    if "native2" in args.modes:
        other = trace(args.logs, args.second_run, args.requests, args.problems, args.texts, args.max_tokens)
        report["second_trace"] = {"run": args.second_run, "kinds": [r["type"] for r in other]}
        report["wspserve_two_sessions"] = run_native([requests, other], args.logs, join=not args.no_join, gather_s=args.gather)
        print("wspserve x2 sessions", round(report["wspserve_two_sessions"]["wall_s"], 1), "s", flush=True)
    if "naive2" in args.modes:   # the second trace alone under the naive design, to compare the pair
        other = trace(args.logs, args.second_run, args.requests, args.problems, args.texts, args.max_tokens)
        report["naive_second_trace"] = run_naive(other, args.logs)
        print("naive second trace", round(report["naive_second_trace"]["wall_s"], 1), "s", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
