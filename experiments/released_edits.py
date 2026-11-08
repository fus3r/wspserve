"""The released Experiment 8 edits, rebuilt from the authors' bundles, and what p50c does.

    python experiments/released_edits.py --logs <lm_understand-logs>/exp8_math

For each tensor p50c edits, finds the direction u of the change, checks that every edited
column's component along u is multiplied by the same factor (so the edit is W - c uu^T W on
those columns), and reports how much of the change lies along u.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from wspdelta import bf16  # noqa: E402
from wspdelta.safetensors_raw import SafeTensors  # noqa: E402


def rebuild_from_bundle(base, tensor, bundle, i):
    """The authors' apply_edit.py, with the same arithmetic: BF16 base, float64 low-rank sum,
    rounding back to BF16, then the listed exact values."""
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


def final_edit_bits(base_source, logs, run):
    manifest = json.loads((logs / f"runs/{run}/edit/manifest.json").read_text())
    with np.load(logs / f"runs/{run}/edit/edit.npz") as bundle:
        return {t["name"]: rebuild_from_bundle(base_source.bf16(t["name"]), t, bundle, i)
                for i, t in enumerate(manifest["tensors"])}


def load_final_edit(engine, logs, run):
    """Edited tensors as float32 arrays, ready for Engine.register_float32."""
    return {name: bf16.to_f32(bits) for name, bits in final_edit_bits(engine.st, logs, run).items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    from wspserve.engine import REPO as QWEN, REVISION

    st = SafeTensors(hf_hub_download(QWEN, "model.safetensors-00001-of-00001.safetensors", revision=REVISION))
    directions = {}
    for name, edited in final_edit_bits(st, args.logs, "p50c").items():
        W, E = bf16.to_f64(st.bf16(name)), bf16.to_f64(edited)
        cols = np.flatnonzero(np.any(W != E, axis=0))
        delta = (E - W)[:, cols]
        u = np.linalg.svd(delta, full_matrices=False)[0][:, 0]
        ratio = (u @ E[:, cols]) / (u @ W[:, cols])
        outside = np.linalg.norm(delta - np.outer(u, u @ delta)) ** 2 / np.linalg.norm(delta) ** 2
        directions[name] = u
        print(f"{name}: {len(cols)} columns, component along u multiplied by {np.median(ratio):.3f} "
              f"(interquartile {np.percentile(ratio, 25):.3f} to {np.percentile(ratio, 75):.3f}), "
              f"{100 * (1 - outside):.4f}% of the change along u")
    a, b = directions.values()
    print(f"cosine between the two directions: {abs(a @ b):.3f}")


if __name__ == "__main__":
    main()
