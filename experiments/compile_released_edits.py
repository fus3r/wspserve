"""Compile the released final WSP edits into typed, bit-exact representations.

For every tensor of every run whose base checkpoint is available, this script
1. reads the pinned base tensor and checks its published SHA-256,
2. rebuilds the evaluated edited tensor from the authors' bundle and checks that hash,
3. recovers the smallest exact representation (ChannelScale, LowRank + fixups, or the
   authors' SparseOverlay) from the two BF16 tensors alone, and checks it rebuilds the
   edited hash bit for bit,
4. for p10c, also rebuilds from the authors' own pre-rounding rank-one factors,
5. writes PEFT LoRA adapters for the runs and a JSON report.

Base checkpoints: Qwen/Qwen3.5-0.8B is public. google/gemma-3-270m-it is gated on Hugging
Face, its runs are reported as unavailable unless the local account has accepted the
license.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wspdelta import bf16, serving_cost  # noqa: E402
from wspdelta.delta import LowRank, SparseOverlay  # noqa: E402
from wspdelta.export import peft_adapter  # noqa: E402
from wspdelta.serialize import save as save_compiled  # noqa: E402
from wspdelta.safetensors_raw import SafeTensors  # noqa: E402
from wspdelta.structure import analyze  # noqa: E402

OUT = ROOT / "results/released_edits_compiled.json"
BASES = {"qwen": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
         "gemma": ("google/gemma-3-270m-it", "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3")}
RUNS = {"p10c": ("qwen", "runs/p10c"), "p25c": ("qwen", "runs/p25c"), "p50c": ("qwen", "runs/p50c"),
        "g05b": ("gemma", "gemma_followup/runs/g05b"), "g10b": ("gemma", "gemma_followup/runs/g10b"),
        "g25b": ("gemma", "gemma_followup/runs/g25b"), "g50b": ("gemma", "gemma_followup/runs/g50b")}


def base_file(key):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    repo, revision = BASES[key]
    try:
        try:
            return Path(hf_hub_download(repo, "model.safetensors", revision=revision)), None
        except Exception as exc:  # Qwen ships one shard under a different name.
            if isinstance(exc, GatedRepoError):
                raise
            return Path(hf_hub_download(repo, "model.safetensors-00001-of-00001.safetensors",
                                        revision=revision)), None
    except GatedRepoError as exc:
        return None, f"gated repository not accessible with the local Hugging Face account ({type(exc).__name__})"


def rebuild_from_bundle(base, tensor, bundle, i):
    """The authors' apply_edit.py, reimplemented with the exact same arithmetic."""
    if tensor["form"] == "lowrank":
        w = bf16.to_f64(base)
        a, b = bundle[f"t{i}_a"], bundle[f"t{i}_b"]
        for k in range(a.shape[1]):
            w = w + np.outer(a[:, k], b[k])
        edited = bf16.from_f64(w).reshape(-1)
    else:
        edited = base.copy().reshape(-1)
    edited[bundle[f"t{i}_idx"]] = bundle[f"t{i}_val"]
    return edited.reshape(base.shape)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=list(RUNS), choices=list(RUNS))
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    selected, logs, out = args.runs, args.logs, args.out
    adapters = out.parent / "adapters"
    compiled = out.parent / "compiled"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"logs_revision": "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f", "bases": BASES,
              "scope": "Released final edits only. Every representation is checked to rebuild the published "
                       "edited SHA-256 bit for bit; no model is run here.", "runs": {}}
    if out.exists():  # keep runs compiled earlier, recompile only the selected ones
        report["runs"] = json.loads(out.read_text()).get("runs", {})
    files = {}
    for run, (key, rel) in RUNS.items():
        if run not in selected:
            continue
        if key not in files:
            files[key] = base_file(key)
        path, unavailable = files[key]
        manifest = json.loads((logs / rel / "edit/manifest.json").read_text())
        if path is None:
            report["runs"][run] = {"status": "base unavailable", "reason": unavailable,
                                   "tensors": len(manifest["tensors"])}
            print(run, "skipped:", unavailable)
            continue
        st = SafeTensors(path)
        started = time.time()
        rows, deltas, bases = [], {}, {}
        with np.load(logs / rel / "edit/edit.npz", allow_pickle=False) as bundle:
            for i, tensor in enumerate(manifest["tensors"]):
                base = st.bf16(tensor["name"])
                assert list(base.shape) == tensor["shape"]
                assert bf16.sha256(base) == tensor["base_sha256"], f"{run} {tensor['name']}: base hash"
                edited = rebuild_from_bundle(base, tensor, bundle, i)
                assert bf16.sha256(edited) == tensor["edited_sha256"], f"{run} {tensor['name']}: edited hash"
                best, options, details = analyze(base, edited)
                for option in options:
                    assert bf16.sha256(option.materialize(base)) == tensor["edited_sha256"]
                row = {"tensor": tensor["name"], "shape": tensor["shape"], "authors_form": tensor["form"],
                       "edited_hash_verified": True, **details, "serving_cost_per_token": serving_cost(best, base.shape),
                       "dense_matmul_madds_per_token": int(base.shape[0] * base.shape[1])}
                if tensor["form"] == "lowrank":
                    authors = LowRank(bundle[f"t{i}_a"], bundle[f"t{i}_b"],
                                      bundle[f"t{i}_idx"].astype(np.int64), bundle[f"t{i}_val"])
                    assert bf16.sha256(authors.materialize(base)) == tensor["edited_sha256"]
                    row["authors_factors"] = {"rank": int(authors.rank), "fixups": int(len(authors.fixup_index)),
                                              "bytes": authors.nbytes()}
                rows.append(row)
                deltas[tensor["name"]] = best
                bases[tensor["name"]] = base
        compiled.mkdir(parents=True, exist_ok=True)
        save_compiled(compiled / f"{run}.npz", deltas)
        adapter = None
        if not any(isinstance(d, SparseOverlay) for d in deltas.values()):
            adapter = peft_adapter(deltas, bases, adapters / run, base_model=BASES[key][0], revision=BASES[key][1])
        kinds = sorted({r["smallest_exact"] for r in rows})
        report["runs"][run] = {"status": "compiled", "tensors": len(rows), "smallest_exact_kinds": kinds,
                               "total_bytes_smallest": int(sum(min(o["bytes"] for o in r["representations"])
                                                               for r in rows)),
                               "total_bytes_authors_bundle": int((logs / rel / "edit/edit.npz").stat().st_size),
                               "dense_bytes_bf16": int(sum(np.prod(r["shape"]) * 2 for r in rows)),
                               "peft_adapter": adapter, "seconds": round(time.time() - started, 1), "rows": rows}
        print(run, kinds, "tensors", len(rows), "smallest bytes", report["runs"][run]["total_bytes_smallest"],
              "adapter", adapter)
    report["runs"] = {run: report["runs"][run] for run in RUNS if run in report["runs"]}
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
