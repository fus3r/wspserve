"""Reuse the local scoring protocol, never import the third-party snapshot."""
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np


def protocol():
    scripts = Path(__file__).resolve().parents[1]
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return importlib.import_module("minatar_protocol")


def _score_chunk(args):
    game, weights, episodes, offset = args
    scores, lengths = [], []
    for i in range(episodes):
        s, n = protocol().evaluate(game, weights, 1, offset=offset+i)
        scores.extend(s)
        lengths.extend(n)
    return scores, lengths


def score(game, weights, episodes=1000, offset=0, workers=8):
    if episodes < 1 or offset < 0 or not 1 <= workers <= 8:
        raise ValueError("episodes must be positive, offset nonnegative, and workers in 1..8")
    if game not in protocol().CAPS:
        raise ValueError(f"unknown game {game!r}")
    started = time.monotonic()
    for variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[variable] = "1"
    # Small chunks distribute long episodes without changing either seed stream.
    jobs = [(game, weights, min(10, episodes-i), offset+i) for i in range(0, episodes, 10)]
    if workers == 1:
        chunks = list(map(_score_chunk, jobs))
    else:
        with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            chunks = pool.map(_score_chunk, jobs)
    scores = [s for chunk, _ in chunks for s in chunk]
    lengths = [s for _, chunk in chunks for s in chunk]
    mean = float(np.mean(scores))
    return {"game": game, "episodes": episodes, "seed_offset": offset, "raw_score": mean,
            "score_std": float(np.std(scores)), "normalized": min(max(mean, 0), protocol().CAPS[game]) / protocol().CAPS[game],
            "mean_episode_length": float(np.mean(lengths)), "seconds": round(time.monotonic()-started, 3),
            "scores": scores, "lengths": lengths}
