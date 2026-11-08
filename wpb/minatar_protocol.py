"""WeightProgramBench's held-out MinAtar games: trivial and hand-compiled baselines.

This is an independent reimplementation of the benchmark's documented scoring protocol
(dataset README and runtime/verify.py at revision cb8d15ea), written here so that no
downloaded file is executed: 1,000 episodes, environment seed 1729 + i, action RNG
RandomState(7919 + i), sticky actions 0.1, difficulty ramping off, at most 10,000
actions, the benchmark's channel mapping, a 600-32-32-6 tanh network with softmax
sampling, and reward = clip(mean score, 0, cap) / cap. Only the PyPI MinAtar 1.0.15
games run. Every policy is also written out as the ``edits`` block a model would
submit, so the official scorer can re-grade it.

Use NumPy 2.3.4 and MinAtar 1.0.15.
"""
import argparse
import sys
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/wpb/baselines.json"
SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}
# Copied from the benchmark's runtime/verify.py (the specification, not imported).
CHANNEL_GROUPS = {
    "asterix": ((0,), (1,), (3,), (), (2,), ()),
    "breakout": ((0,), (), (3,), (1,), (2,), ()),
    "freeway": ((0,), (1,), (), (), (2, 3, 4, 5, 6), ()),
    "seaquest": ((0, 1), (5, 6, 4), (5, 6, 9), (2,), (3, 1), (7, 8)),
    "space_invaders": ((0,), (1, 5), (1,), (4,), (2, 3), ()),
}
CAPS = {"asterix": 1000.0, "breakout": 1000.0, "freeway": 100.0, "seaquest": 1000.0, "space_invaders": 1000.0}
NOOP, LEFT, UP, RIGHT, DOWN, FIRE = range(6)


def cell(row, col, channel):
    """Index of one input in the flattened 10 x 10 x 6 observation."""
    return (row * 10 + col) * 6 + channel


def zeros():
    return {name: np.zeros(shape, dtype=np.float32) for name, shape in SHAPES.items()}


# ---------------------------------------------------------------- policies

def constant(action):
    w = zeros()
    w["b3"][action] = 50.0
    return w


def breakout_tracker():
    """Three hidden units: paddle moves toward the ball's column.

    Unit 1 computes tanh(2 * (ball_col - paddle_col)) from the ball channel (3) and the
    paddle channel (0), unit 2 passes it on, the output turns it into left/right, with a
    noop preference when they are aligned.
    """
    w = zeros()
    for r in range(10):
        for c in range(10):
            w["W1"][cell(r, c, 3), 0] += 2.0 * c
            w["W1"][cell(r, c, 0), 0] -= 2.0 * c
    w["W2"][0, 0] = 3.0
    w["W3"][0, RIGHT], w["W3"][0, LEFT] = 20.0, -20.0
    w["b3"][NOOP] = 5.0
    return w


def breakout_lead(lead=2.0):
    """One unit: move toward ball_col + lead * (ball_col - trail_col), the ball's heading.

    The trail channel (4) marks the ball's previous cell, so the difference is its
    horizontal velocity. Still linear in the inputs, so one tanh unit suffices.
    """
    w = zeros()
    for r in range(10):
        for c in range(10):
            w["W1"][cell(r, c, 3), 0] += 2.0 * (1.0 + lead) * c
            w["W1"][cell(r, c, 4), 0] -= 2.0 * lead * c
            w["W1"][cell(r, c, 0), 0] -= 2.0 * c
    w["W2"][0, 0] = 3.0
    w["W3"][0, RIGHT], w["W3"][0, LEFT] = 20.0, -20.0
    w["b3"][NOOP] = 5.0
    return w


def freeway_crosser(look=(3, 4, 5)):
    """Up unless a car occupies the row just above the chicken, in columns ``look``.

    One first-layer unit per (chicken row, column): it fires when the chicken (channel 0)
    is in that row's column 4 and a car (channel 1) is in the row above. A second-layer
    unit ORs them, the output prefers noop when it fires and up otherwise.
    """
    w = zeros()
    j = 0
    for row in range(1, 10):
        for col in look:
            if j >= 32:
                break
            w["W1"][cell(row, 4, 0), j] = 4.0
            w["W1"][cell(row - 1, col, 1), j] = 4.0
            w["b1"][j] = -6.0          # both present: tanh(2) ~ +0.96, else <= tanh(-2) ~ -0.96
            j += 1
    used = j
    w["W2"][:used, 0] = 2.0
    w["b2"][0] = 2.0 * (used - 1)      # OR: +1 if any unit fires, about -1 otherwise
    w["W3"][0, NOOP], w["W3"][0, UP] = 10.0, -10.0
    w["b3"][UP] = 10.0
    return w


POLICIES = {
    "zero": ("all", lambda: zeros()),
    **{f"const_{a}": ("all", (lambda a=a: constant(a))) for a in range(6)},
    "breakout_tracker": ("breakout", breakout_tracker),
    **{f"breakout_lead{l:g}": ("breakout", (lambda l=l: breakout_lead(l))) for l in (1, 2, 3, 4, 6, 9)},
    "freeway_crosser": ("freeway", freeway_crosser),
    **{f"freeway_look{''.join(map(str, look))}": ("freeway", (lambda look=look: freeway_crosser(look)))
       for look in ((4,), (3, 4), (4, 5), (2, 3, 4, 5, 6), (3, 4, 5, 6), (2, 3, 4, 5))},
}


# ---------------------------------------------------------------- scoring

def observation(game, state):
    channels = np.zeros((10, 10, 6), dtype=np.float32)
    for out, sources in enumerate(CHANNEL_GROUPS[game]):
        if sources:
            channels[:, :, out] = np.any(state[:, :, sources], axis=2)
    return channels.reshape(600)


def logits(w, x):
    h1 = np.tanh(x @ w["W1"] + w["b1"])
    h2 = np.tanh(h1 @ w["W2"] + w["b2"])
    return h2 @ w["W3"] + w["b3"]


def sample(action_logits, rng):
    shifted = action_logits.astype(np.float64) - np.max(action_logits)
    p = np.exp(shifted)
    p /= p.sum()
    return int(rng.choice(6, p=p))


def evaluate(game, w, episodes, offset=0):
    """offset = 0 is the benchmark's protocol, tuning uses a disjoint seed range."""
    from minatar import Environment

    scores, lengths = [], []
    for episode in range(episodes):
        env = Environment(game, sticky_action_prob=0.1, difficulty_ramping=False)
        env.seed(1729 + offset + episode)
        rng = np.random.RandomState(7919 + offset + episode)
        env.reset()
        score = 0.0
        for step in range(10_000):
            reward, terminal = env.act(sample(logits(w, observation(game, env.state())), rng))
            score += float(reward)
            if terminal:
                break
        scores.append(score)
        lengths.append(step + 1)
    return scores, lengths


def edits_block(w):
    """The submission a model would write: one assignment per nonzero parameter."""
    lines = []
    for name, array in w.items():
        for index in zip(*np.nonzero(array)):
            target = ", ".join(str(int(i)) for i in index)
            lines.append(f"{name}[{target}] = {float(array[index])!r}")
    return "```edits\n" + "\n".join(lines) + "\n```\n"


def job(args):
    game, policy, episodes, offset = args
    w = POLICIES[policy][1]()
    started = time.time()
    scores, lengths = evaluate(game, w, episodes, offset)
    mean = float(np.mean(scores))
    return {"game": game, "policy": policy, "episodes": episodes, "seed_offset": offset, "raw_score": mean,
            "score_std": float(np.std(scores)), "normalized": min(max(mean, 0.0), CAPS[game]) / CAPS[game],
            "mean_episode_length": float(np.mean(lengths)),
            "nonzero_parameters": int(sum(np.count_nonzero(a) for a in w.values())),
            "seconds": round(time.time() - started, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", nargs="*", default=list(CAPS))
    parser.add_argument("--policies", nargs="*", default=list(POLICIES))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="0 = official seeds; tuning runs use e.g. 100000 and are written separately")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or (OUT if args.seed_offset == 0 else OUT.with_name("baselines_tuning.json"))
    policy_dir = out.parent / "policies"
    policy_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(g, p, args.episodes, args.seed_offset) for g in args.games for p in args.policies
            if POLICIES[p][0] in ("all", g)]
    for g, p, _, _ in jobs:
        (policy_dir / f"{g}__{p}.txt").write_text(edits_block(POLICIES[p][1]()))
    rows = []
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(job, jobs):
            rows.append(row)
            print(f"{row['game']:>15} {row['policy']:>17} raw={row['raw_score']:8.2f} norm={row['normalized']:.4f} "
                  f"len={row['mean_episode_length']:7.1f} nz={row['nonzero_parameters']} {row['seconds']}s", flush=True)
    previous = json.loads(out.read_text())["rows"] if out.exists() else []
    done = {(r["game"], r["policy"], r["episodes"]) for r in rows}
    merged = [r for r in previous if (r["game"], r["policy"], r["episodes"]) not in done] + rows
    out.write_text(json.dumps({"protocol": "WeightProgramBench scoring, reimplemented (see docstring)",
                               "seed_offset": args.seed_offset,
                               "dataset_revision": "cb8d15ea0d2c1970f6d8b590c7b7fcb774acd426",
                               "rows": sorted(merged, key=lambda r: (r["game"], r["policy"], r["episodes"]))},
                              indent=2) + "\n")


if __name__ == "__main__":
    main()
