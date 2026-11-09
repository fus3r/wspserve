# Reproducing the WeightProgramBench baselines

The results use `neilkale/WeightProgramBench` at dataset revision
`cb8d15ea0d2c1970f6d8b590c7b7fcb774acd426`, whose frozen runtime comes from
MinAtar Synth revision `4038830175c4201ca10e004ed4441985f2acddf2`.
The scorer checks the pinned `runtime/source.json` digest, then its scorer, task and
game-file hashes before importing the runtime. No sibling `REVISION` file is needed.

Run these commands from the repository root. The WPB replay only needs NumPy and
MinAtar, so it can use a separate environment from the inference experiments:

```sh
python3 -m venv .venv-wpb
.venv-wpb/bin/python -m pip install huggingface_hub 'numpy==2.3.4'
.venv-wpb/bin/python -m pip install --no-deps 'MinAtar==1.0.15'
.venv-wpb/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "neilkale/WeightProgramBench",
    repo_type="dataset",
    revision="cb8d15ea0d2c1970f6d8b590c7b7fcb774acd426",
    allow_patterns=["runtime/**", "README.md", "THIRD_PARTY_NOTICES.md"],
    local_dir=".data/weightprogrambench",
)
PY
```

The downloaded runtime is the benchmark authors' Python scorer and game code.
Upstream attribution and notices remain with that snapshot. The runtime uses environment
seeds `1729 + i`, action seeds `7919 + i`, 10% sticky actions and a 10,000-action limit.

## Held-out games

The official scorer evaluates 35 zero/constant policies and 14 hand-written rules on
the five held-out MinAtar games, 1,000 episodes per policy:

```sh
.venv-wpb/bin/python wpb/baselines.py --wpb .data/weightprogrambench \
  --split test --answers results/wpb/policies --workers 4 \
  --out .data/reproduced/official_baselines.json
```

Compare with [official_baselines.json](../results/wpb/official_baselines.json).
For the four scores used in the post: Breakout zero is 0.482, Breakout `breakout_lead1`
is 10.402, Freeway `const_2` is 17.043 and Freeway `freeway_look4` is 44.848.
`baselines.json` is the earlier independent MinAtar protocol reimplementation;
`official_baselines.json` records the official-scorer check of the same policies.
Compare scores and episode counts; elapsed times depend on the machine.

## Training games

The training survey used 100 episodes per policy, rather than the benchmark's default
1,000. Seven trivial policies across 317 games produce 2,219 scores:

```sh
.venv-wpb/bin/python wpb/baselines.py --wpb .data/weightprogrambench \
  --split train --episodes 100 --workers 4 \
  --out .data/reproduced/official_baselines_train100.json
```

The recorded survey is [official_baselines_train100.json](../results/wpb/official_baselines_train100.json).
Three recorded outcomes are runtime errors assigned zero by the official scorer:
`time_pilot` with `const_2` and `zero`, and `tutankham` with `zero`. Their `error` fields
are preserved; a missing raw score is not a measured zero return. The summary uses the
scorer's normalized outcomes, including those error outcomes.

Rebuild its [per-game summary](../results/wpb/train_trivial.json) from a replay with:

```sh
.venv-wpb/bin/python - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

root = Path(".data/reproduced")
rows = json.loads((root / "official_baselines_train100.json").read_text())["rows"]
games = defaultdict(list)
for row in rows:
    games[row["task"]].append(row)
summary = {}
for game, policies in sorted(games.items()):
    best = max(policies, key=lambda row: row["normalized"])
    zero = next(row for row in policies if row["policy"] == "zero")
    summary[game] = {"best_trivial_normalized": best["normalized"], "policy": best["policy"],
                     "zero_normalized": zero["normalized"]}
(root / "train_trivial.json").write_text(json.dumps(summary, indent=2) + "\n")
print([game for game, row in summary.items() if row["best_trivial_normalized"] == 1.0])
PY
```

This gives eight saturated games: `double_dunk`, `fishing_derby`, `hyper_laser`, `kite`,
`robot_tank`, `stomping_bubbles`, `tt_fence` and `wiper`. These are 100-episode estimates.

For a short command check, use `--split test --episodes 1 --answers results/wpb/policies`
with a separate `--out` path. That checks the scorer and report-writing path; it does
not reproduce the published 1,000-episode scores.
