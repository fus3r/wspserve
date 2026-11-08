"""Figures for the write-up, in plain matplotlib.

accuracy.png: MATH-500 accuracy on the 200 problems with released texts (first sample), for
the unedited model, the unedited model under the stop rule, and the p50c edit, with 95%
bootstrap intervals over problems.

loop_direction.png: cosine between the residual stream entering block 23 and p50c's layer-19
direction, along 24 looping traces of the unedited model aligned at the first loop flag,
under the original, p50c and p25c weights (teacher forcing). Bands are 95% bootstrap
intervals over traces.

gpu_time.png: wall-clock time of each released session, from the authors' logged service times.

gpu_replay.png: wall-clock time to replay 1, 2, 4 and 8 released sessions at once on one H100
with wspserve, and the tokens they generated.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(0)


def boot_ci(values, n=10_000):
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def accuracy():
    rule = {r["id"]: r["correct"] for r in map(json.loads, open(ROOT / "results/stop_and_answer_unedited.jsonl"))}
    une = {r["id"]: bool(r["correct_first"]) for r in map(json.loads, open(ROOT / "results/outcomes_unedited.jsonl"))}
    p50 = {r["id"]: bool(r["correct_first"]) for r in map(json.loads, open(ROOT / "results/outcomes_p50c.jsonl"))}
    ids = sorted(rule)
    rows = [("unedited", [une[i] for i in ids]),
            ("unedited, stopped at first loop", [rule[i] for i in ids]),
            ("p50c (edited weights)", [p50[i] for i in ids])]
    fig, ax = plt.subplots(figsize=(6.4, 2.2), dpi=200)
    for y, (label, v) in enumerate(reversed(rows)):
        m = 100 * np.mean(v)
        lo, hi = 100 * boot_ci(v)
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", color="black", capsize=3, markersize=5)
        ax.text(hi + 1.5, y, f"{m:.1f}%", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(f"MATH-500 accuracy, same {len(ids)} problems, first sample (%)")
    ax.grid(axis="x", alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy.png")
    print("accuracy.png", [(r[0], sum(r[1])) for r in rows])


def loop_direction():
    series = defaultdict(lambda: defaultdict(dict))   # variant -> bin -> trace -> cosine
    for r in csv.DictReader(open(ROOT / "results/loop_direction_timeseries.csv")):
        if r["layer_hidden_index"] == "23" and r["direction"] == "p50c_L19_mlp":
            center = (int(r["bin_start"]) + int(r["bin_end"])) / 2
            key = r["variant"] if r["group"] == "loop" else "finish_" + r["variant"]
            series[key][center][r["id"]] = float(r["cosine"])
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
    styles = {"original": ("unedited, looping traces", "black", "-"), "p50c": ("p50c, same traces", "tab:blue", "-"),
              "p25c": ("p25c, same traces", "tab:orange", "--"),
              "finish_original": ("unedited, traces that finish (aligned at end of thinking)", "gray", ":")}
    for variant, (label, color, ls) in styles.items():
        xs = sorted(series[variant])
        means = [np.mean(list(series[variant][x].values())) for x in xs]
        cis = [boot_ci(list(series[variant][x].values())) for x in xs]
        ax.plot(xs, means, ls, color=color, label=label, linewidth=1.5)
        ax.fill_between(xs, [c[0] for c in cis], [c[1] for c in cis], color=color, alpha=0.15, linewidth=0)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("tokens relative to the first loop flag (or the end of thinking)")
    ax.set_ylabel("cosine with loop direction\n(residual stream, block 23)")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.62))
    ax.grid(alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "loop_direction.png")
    n = len(next(iter(series["original"].values())))
    print("loop_direction.png", n, "traces")


def gpu_time():
    """Wall-clock time of each released session, split into GPU busy on passes that completed,
    GPU busy on generation passes that failed while starting the engine, and idle time while the
    editor thinks. Documented outages are removed. From the authors' logged service times."""
    d = json.load(open(ROOT / "results/workload.json"))["sessions"]
    order = ["p10c", "p25c", "p50c", "g05b", "g10b", "g25b", "g50b"]
    names = {"p10c": "Qwen, 10 passes", "p25c": "Qwen, 25 passes", "p50c": "Qwen, 50 passes",
             "g05b": "Gemma, 5 passes", "g10b": "Gemma, 10 passes", "g25b": "Gemma, 25 passes", "g50b": "Gemma, 50 passes"}
    fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=200)
    for y, k in enumerate(reversed(order)):
        s_ = d[k]
        wall = s_["span_h"] - s_.get("outage_gaps_h", 0.0)
        failed = s_["waste"]["failed_attempt_service_h"]
        done = s_["busy_h"] - failed
        idle = wall - s_["busy_h"]
        left = 0.0
        for value, color, label in ((done, "tab:blue", "GPU busy, pass completed"),
                                    (failed, "tab:red", "GPU busy, engine failed to start"),
                                    (idle, "0.85", "GPU idle, editor thinking")):
            ax.barh(y, 100 * value / wall, left=left, color=color, height=0.6,
                    label=label if y == 0 else None, edgecolor="white", linewidth=0.5)
            left += 100 * value / wall
        ax.text(101, y, f"{wall:.1f} h", va="center", fontsize=7, color="0.3")
    ax.set_yticks(range(len(order)), [names[k] for k in reversed(order)], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the session's wall-clock time (%)")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.45, 1.0))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "gpu_time.png")
    print("gpu_time.png", {k: round(d[k]["dedicated_busy_fraction"], 2) for k in order})


def gpu_replay():
    """Released sessions replayed at once on one GPU: wall-clock time and tokens generated."""
    runs = json.load(open(ROOT / "results/gpu_bench.json"))["runs"]
    ks = [r["sessions"] for r in runs]
    minutes = [r["wall_s"] / 60 for r in runs]
    x = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(6.4, 2.4), dpi=200)
    ax.bar(x, minutes, width=0.55, color="tab:blue")
    for xi, m, r in zip(x, minutes, runs):
        ax.text(xi, m + 0.1, f"{m:.1f} min, {r['generated_tokens'] / 1000:.0f}k tokens", ha="center",
                va="bottom", fontsize=7, color="0.2")
    ax.set_xticks(x, [f"{k} session{'s' if k > 1 else ''}" for k in ks], fontsize=8)
    ax.set_ylabel("wall-clock minutes", fontsize=8)
    ax.set_ylim(0, max(minutes) * 1.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "gpu_replay.png")
    print("gpu_replay.png", dict(zip(ks, [round(m, 2) for m in minutes])))


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description="Rebuild the figures from results.").parse_args()
    accuracy()
    loop_direction()
    gpu_time()
    gpu_replay()
