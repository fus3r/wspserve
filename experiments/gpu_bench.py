"""How many released WSP sessions one GPU serves at once.

Each session replays the first requests of a released Experiment 8 session (p50c, p25c or
p10c), with its own problems (copies of a session take disjoint slices of the problem lists, so
the base-sample memo is never shared between sessions), the released p50c edit as its edited
version, n = 1 and --max-tokens per answer. All sessions run against one wspserve process.

    python experiments/gpu_bench.py --logs <logs>/exp8_math --sessions 1 2 4 8

Writes results/gpu_bench.json: per run, wall time, generated tokens, decode steps, every
request's latency, and probe waits.

--paired N instead times one paired generation pass on the first N/2 GSM8K and N/2 MATH
practice problems with the loop stop off, the pass vllm_lora.py runs through vLLM:

    python experiments/gpu_bench.py --logs <logs>/exp8_math --sessions --paired 64 --out results/paired_pass.json
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.replay import p50c_float32  # noqa: E402

RUNS = ["p50c", "p25c", "p10c"]


def session_trace(logs, run, count, problems, texts, max_tokens, offset):
    rows = [json.loads(line) for line in (logs / f"runs/{run}/tool_calls.jsonl").read_text().splitlines()]
    out = []
    for r in rows:
        if r["kind"] != "forward" or not r["result"].get("ok"):
            continue
        req = dict(r["request"])
        if req["type"] == "generate":
            items = req.get("problems") or []
            if len(items) < offset + problems:
                continue
            out.append({"type": "generate", "model": req.get("model", "both"),
                        "problems": items[offset:offset + problems], "n": 1, "max_tokens": max_tokens})
        else:
            key = "chats" if req.get("chats") else "texts"
            item = {"type": "teacher_forced", "model": req.get("model", "original"),
                    key: req[key][:texts], "return": {"token_logprobs": True}}
            if key == "chats":
                item["enable_thinking"] = req.get("enable_thinking", True)
            out.append(item)
        if len(out) >= count:
            break
    return out


def run(svc, edits, traces):
    timings, passes = {}, {}

    def replay(sid, requests):
        s = svc.session(sid)
        s.edits = dict(edits)
        s.version = svc.engine.register_float32(s.edits)
        start, res = time.time(), []
        for req in requests:
            t = time.time()
            r = svc.forward(sid, req)
            res.append({"type": req["type"], "s": round(time.time() - t, 2), "ok": bool(r.get("ok")),
                        "wait_s": r.get("wait_s"), "memo_hits": r.get("memo_hits")})
        timings[sid] = round(time.time() - start, 1)
        passes[sid] = res

    before = dict(svc.engine.stats)
    threads = [threading.Thread(target=replay, args=(f"s{k}", reqs)) for k, reqs in enumerate(traces)]
    started = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - started
    stats = {k: svc.engine.stats.get(k, 0) - before.get(k, 0) for k in svc.engine.stats}
    probes = [p["wait_s"] for ps in passes.values() for p in ps if p["type"] == "teacher_forced" and p["wait_s"] is not None]
    return {"sessions": len(traces), "wall_s": round(wall, 1), "per_session_s": timings,
            "generated_tokens": stats.get("generated_tokens", 0), "decode_steps": stats.get("decode_steps", 0),
            "tokens_per_s": round(stats.get("generated_tokens", 0) / wall, 1),
            "probe_wait_s_max": max(probes) if probes else None, "passes": passes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--sessions", type=int, nargs="*", default=[1, 2, 4, 8])
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--problems", type=int, default=8)
    parser.add_argument("--texts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--paired", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO / "results/gpu_bench.json")
    args = parser.parse_args()

    import torch

    from wspserve.server import Service

    svc = Service(max_rows=args.max_rows)
    edits = p50c_float32(svc.engine, args.logs)
    report = {"device": str(svc.engine.device),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "settings": {k: v for k, v in vars(args).items() if k not in ("logs", "out")}, "runs": []}
    # Warm-up on other problems (kernel compilation and autotuning), not reported.
    run(svc, edits, [session_trace(args.logs, RUNS[0], 3, args.problems, args.texts, 64, offset=100)])
    for k in args.sessions:
        traces = [session_trace(args.logs, RUNS[i % len(RUNS)], args.requests, args.problems, args.texts,
                                args.max_tokens, offset=(i // len(RUNS)) * args.problems) for i in range(k)]
        svc.memo.clear()
        result = run(svc, edits, traces)
        report["runs"].append(result)
        print(json.dumps({x: result[x] for x in ("sessions", "wall_s", "generated_tokens", "tokens_per_s",
                                                  "probe_wait_s_max")}), flush=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.paired:
        report["paired_pass"] = paired_pass(svc, edits, args.logs / "practice", args.paired, args.max_tokens)
        print(json.dumps(report["paired_pass"]), flush=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


def paired_pass(svc, edits, practice, count, max_tokens):
    ids = []
    for name in ("gsm8k_train.jsonl", "math_train.jsonl"):
        lines = [l for l in (practice / name).read_text().split("\n") if l.strip()]
        ids += [json.loads(l)["id"] for l in lines[: count // 2]]
    svc.memo.clear()
    svc.loop_stop = False
    s = svc.session("paired")
    s.edits = dict(edits)
    s.version = svc.engine.register_float32(s.edits)
    before = dict(svc.engine.stats)
    started = time.time()
    r = svc.forward("paired", {"type": "generate", "model": "both", "problems": ids, "n": 1, "max_tokens": max_tokens})
    elapsed = time.time() - started
    tokens = svc.engine.stats.get("generated_tokens", 0) - before.get("generated_tokens", 0)
    return {"problems": len(ids), "rows": 2 * len(ids), "max_tokens": max_tokens, "loop_stop": False,
            "s": round(elapsed, 1), "generated_tokens": tokens, "tokens_per_s": round(tokens / elapsed, 1),
            "decode_steps": svc.engine.stats.get("decode_steps", 0) - before.get("decode_steps", 0),
            **{half: {"correct": round(r[half]["accuracy"] * r[half]["generations"]),
                      "hit_cap": r[half]["failures"]["cut_at_max_tokens"], "mean_tokens": round(r[half]["mean_tokens"])}
               for half in ("original", "edited")}}


if __name__ == "__main__":
    main()
