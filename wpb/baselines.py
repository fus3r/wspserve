"""Trivial-policy baselines for WeightProgramBench, scored by the benchmark's own grader.

Every policy is submitted exactly as a model would submit it: an ```edits``` block that
the frozen ``runtime/verify.py`` parses with ``ast.literal_eval`` and plays for the
task's 1,000 seeded episodes. Policies:

- ``zero``: no edits. All parameters stay zero, so the softmax is uniform over the six
  actions: the network every task starts from.
- ``const_<a>``: one edit, ``b3[a] = 50.0``. Action a has probability 1 - 5e-21, the
  environment's 10% sticky actions still apply.

The benchmark files are third-party code. They are hash-checked against the dataset's
``runtime/source.json`` before import, they import only NumPy and the standard library.
Use NumPy 2.3.4, MinAtar 1.0.15 and the frozen runtime versions.
"""
import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/wpb/official_baselines.json"
ACTIONS = ["noop", "left", "up", "right", "down", "fire"]
POLICIES = {"zero": "```edits\n```"}
POLICIES.update({f"const_{a}": f"```edits\nb3[{a}] = 50.0\n```" for a in range(6)})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_snapshot(runtime):
    source = json.loads((runtime / "source.json").read_text())
    assert sha256(runtime / "verify.py") == source["verify_sha256"], "verify.py hash"
    assert sha256(runtime / "tasks.jsonl") == source["tasks_sha256"], "tasks.jsonl hash"
    for rel, digest in source["game_sha256"].items():
        assert sha256(runtime / rel) == digest, rel
    return source


def load_verify(runtime):
    sys.path.insert(0, str(runtime))
    spec = importlib.util.spec_from_file_location("wpb_verify", runtime / "verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(job):
    task, policy, episodes, answer, runtime = job
    verify = load_verify(runtime)
    started = time.time()
    normalized, meta = verify.grade_answer(task["game"], task["source"], answer, episodes)
    return {"task": task["name"], "game": task["game"], "split": task["split"], "source": task["source"],
            "policy": policy, "episodes": episodes, "normalized": normalized,
            "raw_score": meta.get("raw_score"), "score_cap": meta.get("score_cap"),
            "mean_episode_length": meta.get("mean_episode_length"), "error": meta.get("error"),
            "seconds": round(time.time() - started, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wpb", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--episodes", type=int, default=None, help="default: each task's own (1,000)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--answers", default=None,
                        help="also grade every <game>__<policy>.txt edits answer in this directory")
    args = parser.parse_args()
    runtime = args.wpb / "runtime"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    source = check_snapshot(runtime)
    tasks = [json.loads(line) for line in (runtime / "tasks.jsonl").read_text().splitlines()]
    tasks = [t for t in tasks if args.split == "all" or t["split"] == args.split]
    jobs = [(t, p, args.episodes or t["episodes"], POLICIES[p], runtime) for t in tasks for p in POLICIES]
    if args.answers:
        by_game = {t["game"]: t for t in tasks}
        for path in sorted(Path(args.answers).glob("*__*.txt")):
            game, policy = path.stem.split("__", 1)
            if game in by_game and policy not in POLICIES:
                jobs.append((by_game[game], policy, args.episodes or by_game[game]["episodes"], path.read_text(), runtime))
    # Longest games first so the pool stays busy.
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        rows = []
        for row in pool.imap_unordered(score, jobs):
            rows.append(row)
            print(f"{row['task']:>16} {row['policy']:>8} norm={row['normalized']:.4f} raw={row['raw_score']} "
                  f"len={row['mean_episode_length']} {row['seconds']}s", flush=True)
    rows.sort(key=lambda r: (r["task"], r["policy"]))
    out = Path(args.out)
    previous = json.loads(out.read_text()) if out.exists() else {"rows": []}
    keep = [r for r in previous["rows"] if (r["task"], r["policy"], r["episodes"]) not in
            {(x["task"], x["policy"], x["episodes"]) for x in rows}]
    report = {"dataset": "neilkale/WeightProgramBench",
              "revision": (SNAPSHOT.parent / "REVISION").read_text().strip(),
              "runtime_source_revision": source["source_revision"],
              "policies": {"compiled": "other names: edits answers graded from --answers",
                           "zero": "no edits: uniform softmax over 6 actions",
                           **{f"const_{a}": f"b3[{a}] = 50.0: always '{ACTIONS[a]}' (plus 10% sticky actions)"
                              for a in range(6)}},
              "rows": sorted(keep + rows, key=lambda r: (r["task"], r["policy"], r["episodes"]))}
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
