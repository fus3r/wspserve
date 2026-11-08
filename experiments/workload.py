"""Characterize the seven released WSP sessions as a serving workload, then project sharing.

Measured from the logs (``t`` is a call's end time, start = t - duration, consistent
for every consecutive pair):
- per session: think time between passes, dedicated-replica busy fraction, pass mix,
- per pass type: service-time distribution, request sizes, token volume,
- the mean residual service time of generation passes, i.e. how long a short probe
  arriving at a random moment waits behind a running pass on a one-request replica.

Projected (labelled as a model, not a measurement): closed-loop replay of N concurrent
sessions on one replica with the logged service and think times, under FIFO without
preemption (NDIF's documented queue) and under processor sharing (an idealized
work-conserving co-batching engine). Service times are the authors' GPU times, nothing
here measures a GPU.
"""
import argparse
import sys
import heapq
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/workload.json"
RUNS = {"p10c": "runs/p10c", "p25c": "runs/p25c", "p50c": "runs/p50c",
        "g05b": "gemma_followup/runs/g05b", "g10b": "gemma_followup/runs/g10b",
        "g25b": "gemma_followup/runs/g25b", "g50b": "gemma_followup/runs/g50b"}
OUTAGE_GAP_S = 2 * 3600  # p50c's README documents one 4.7 h wait for an unreachable editor


def quantiles(values):
    values = sorted(values)
    if not values:
        return None
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return {"n": len(values), "min": values[0], "p50": pick(0.5), "p90": pick(0.9), "max": values[-1],
            "mean": statistics.fmean(values)}


def session(logs, run, rel):
    rows = [json.loads(line) for line in (logs / rel / "tool_calls.jsonl").read_text().splitlines()]
    calls = []
    for r in rows:
        if r["kind"] == "python":
            calls.append(("python", r["t"] - r["s"], r["t"], r))
        elif r["kind"] == "forward":
            calls.append(("forward", r["t"] - r["result"]["s"], r["t"], r))
    calls.sort(key=lambda c: c[1])
    passes, previous_end = [], calls[0][1]
    for kind, start, end, r in calls:
        if kind != "forward":
            continue
        req, res = r["request"], r["result"]
        size = len(req.get("problems") or []) * int(req.get("n") or 1) if req.get("type") == "generate" \
            else len(req.get("texts") or req.get("chats") or [])
        halves = 2 if req.get("model") == "both" else 1
        passes.append({"type": req.get("type"), "model": req.get("model"), "ok": bool(res.get("ok")),
                       "service_s": res["s"], "think_before_s": max(0.0, start - previous_end),
                       "size_per_half": size, "rows": size * halves, "max_tokens": req.get("max_tokens")})
        previous_end = end
    span = calls[-1][2] - calls[0][1]
    gaps = [p["think_before_s"] for p in passes]
    outage = sum(g for g in gaps if g > OUTAGE_GAP_S)
    busy = sum(p["service_s"] for p in passes)
    return passes, {"span_h": span / 3600, "outage_gaps_h": outage / 3600, "busy_h": busy / 3600,
                    "dedicated_busy_fraction": busy / (span - outage)}


def waste(logs, rel):
    """GPU time on failed attempts, and base-model generation that memoization would avoid.

    The base weights never change within a session, so earlier base samples of a problem
    are valid samples of the same distribution. With memoization a session needs, per
    problem, only as many base samples as its largest single request, tokens are
    approximated with each half's reported mean length.
    """
    rows = [json.loads(line) for line in (logs / rel / "tool_calls.jsonl").read_text().splitlines()]
    forward = [r for r in rows if r["kind"] == "forward"]
    failed_s = sum(r["result"]["s"] for r in forward if not r["result"].get("ok"))
    total_s = sum(r["result"]["s"] for r in forward)
    naive_tokens = memo_tokens = 0.0
    most = {}
    for r in forward:
        req, res = r["request"], r["result"]
        if req.get("type") != "generate" or not res.get("ok") or req.get("model") == "edited":
            continue
        half = res["original"] if req.get("model") == "both" else res
        n, problems = int(req.get("n") or 1), req.get("problems") or []
        per_sample = half["mean_tokens"]
        naive_tokens += n * len(problems) * per_sample
        for pid in problems:
            extra = max(0, n - most.get(pid, 0))
            memo_tokens += extra * per_sample
            most[pid] = max(most.get(pid, 0), n)
    gen_tokens = sum(
        sum(h["generations"] * h["mean_tokens"] for h in ([r["result"]["original"], r["result"]["edited"]]
            if r["request"].get("model") == "both" else [r["result"]]))
        for r in forward if r["request"].get("type") == "generate" and r["result"].get("ok"))
    return {"failed_attempt_service_h": failed_s / 3600, "failed_share_of_service": failed_s / total_s,
            "base_generation_tokens": naive_tokens, "base_tokens_avoidable_by_memoization": naive_tokens - memo_tokens,
            "avoidable_share_of_all_generated_tokens": (naive_tokens - memo_tokens) / gen_tokens if gen_tokens else 0.0}


def simulate(traces, sessions, policy, seed=0, horizon_offsets_s=7200.0):
    """Closed-loop replay: each session issues its next pass think_before_s after the last reply."""
    rng = random.Random(seed)
    # Each simulated session replays one released trace, starting at a random offset.
    plan = [traces[i % len(traces)] for i in range(sessions)]
    offsets = [rng.uniform(0, horizon_offsets_s) for _ in range(sessions)]
    pending = []  # (arrival time, session, pass index)
    for s in range(sessions):
        heapq.heappush(pending, (offsets[s] + plan[s][0]["think_before_s"], s, 0))
    now, active, done = 0.0, {}, []  # active: key -> remaining work (s), start times kept separately
    starts, queue = {}, []

    def admit(t, s, k):
        key = (s, k)
        starts[key] = t
        if policy == "ps":
            active[key] = plan[s][k]["service_s"]
        else:
            queue.append(key)

    def finish(key, t):
        s, k = key
        p = plan[s][k]
        done.append({"session": s, "type": p["type"], "service_s": p["service_s"], "response_s": t - starts[key]})
        if k + 1 < len(plan[s]):
            heapq.heappush(pending, (t + plan[s][k + 1]["think_before_s"], s, k + 1))

    running = None  # FIFO: (key, finish time)
    while pending or active or queue or running:
        if policy == "ps":
            if active:
                share = 1.0 / len(active)
                key_min = min(active, key=active.get)
                t_finish = now + active[key_min] / share
            else:
                t_finish = float("inf")
            t_arrival = pending[0][0] if pending else float("inf")
            t_next = min(t_finish, t_arrival)
            if active:
                spent = (t_next - now) / len(active)
                for key in active:
                    active[key] -= spent
            now = t_next
            if t_finish <= t_arrival:
                for key in [k for k, v in active.items() if v <= 1e-9]:
                    del active[key]
                    finish(key, now)
            else:
                t, s, k = heapq.heappop(pending)
                admit(t, s, k)
        else:
            if running is None and queue:
                key = queue.pop(0)
                running = (key, now + plan[key[0]][key[1]]["service_s"])
            t_finish = running[1] if running else float("inf")
            t_arrival = pending[0][0] if pending else float("inf")
            if t_finish <= t_arrival:
                now = t_finish
                finish(running[0], now)
                running = None
            else:
                t, s, k = heapq.heappop(pending)
                now = t
                admit(t, s, k)
    probes = [d["response_s"] for d in done if d["type"] == "teacher_forced"]
    gens = [d["response_s"] / d["service_s"] for d in done if d["type"] == "generate" and d["service_s"] > 0]
    busy = sum(d["service_s"] for d in done)
    return {"sessions": sessions, "policy": policy, "probe_response_s": quantiles(probes),
            "generation_slowdown": quantiles(gens), "makespan_h": now / 3600,
            "server_utilization": busy / now}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"logs_revision": "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f", "sessions": {}, "pass_types": {}}
    traces, all_passes = [], []
    for run, rel in RUNS.items():
        passes, summary = session(args.logs, run, rel)
        thinks = [p["think_before_s"] for p in passes[1:] if p["think_before_s"] <= OUTAGE_GAP_S]
        summary.update({"passes": len(passes), "failed": sum(not p["ok"] for p in passes),
                        "think_time_s": quantiles(thinks), "waste": waste(args.logs, rel)})
        report["sessions"][run] = summary
        # The replayed trace drops the documented outage, keeping every pass and think time.
        traces.append([dict(p, think_before_s=min(p["think_before_s"], OUTAGE_GAP_S)) for p in passes])
        all_passes += [dict(p, run=run, model_family="qwen" if run.startswith("p") else "gemma") for p in passes]
    for family in ("qwen", "gemma"):
        for kind in ("generate", "teacher_forced"):
            ok = [p for p in all_passes if p["ok"] and p["type"] == kind and p["model_family"] == family]
            service = [p["service_s"] for p in ok]
            entry = {"service_s": quantiles(service), "rows": quantiles([p["rows"] for p in ok]),
                     "paired_share": sum(p["model"] == "both" for p in ok) / len(ok) if ok else None}
            if kind == "generate" and service:
                entry["mean_residual_service_s"] = sum(s * s for s in service) / (2 * sum(service))
            report["pass_types"][f"{family}/{kind}"] = entry
    report["projection"] = {
        "assumptions": "Closed-loop replay of the logged traces (outage removed). FIFO: one request at a "
                       "time, no preemption. PS: all active requests share the replica equally, an "
                       "idealized co-batching engine with no overhead. Service times are the authors' GPU's.",
        "runs": [simulate(traces, n, policy, seed=seed) for n in (1, 2, 4, 7, 14) for policy in ("fifo", "ps")
                 for seed in (0,)]}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for run, s in report["sessions"].items():
        w = s["waste"]
        print(run, f"span {s['span_h']:.1f} h, busy {s['busy_h']:.1f} h, dedicated busy fraction "
                   f"{s['dedicated_busy_fraction']:.2f}, think p50 {s['think_time_s']['p50']:.0f} s | failed "
                   f"{w['failed_attempt_service_h']:.2f} h ({w['failed_share_of_service']:.0%}) | memoizable "
                   f"{w['avoidable_share_of_all_generated_tokens']:.0%} of generated tokens")
    for k, v in report["pass_types"].items():
        print(k, {kk: (round(vv["p50"], 1) if isinstance(vv, dict) else vv) for kk, vv in v.items()
                  if vv is not None})
    for r in report["projection"]["runs"]:
        pr = r["probe_response_s"]
        print(f"N={r['sessions']:>2} {r['policy']:>4} probe p50 {pr['p50']:.1f}s p90 {pr['p90']:.1f}s | "
              f"gen slowdown p50 {r['generation_slowdown']['p50']:.2f} | util {r['server_utilization']:.2f} | "
              f"makespan {r['makespan_h']:.1f} h")


if __name__ == "__main__":
    main()
