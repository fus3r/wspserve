"""Editor-side helpers with the Sparks of WSP kernel names, talking to a wspserve server.

    import wspserve.client as W
    w = W.load_original("model.language_model.layers.19.mlp.down_proj.weight")
    W.save_edit(name, w_new)
    W.forward({"type": "generate", "model": "both", ...})

Environment: WSP_SERVER (default http://127.0.0.1:8765), WSP_SESSION (required), WSP_WORK
(the editor's folder, edits are written to WSP_WORK/edited/<name>.npy).
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SERVER = os.environ.get("WSP_SERVER", "http://127.0.0.1:8765")
SESSION = os.environ.get("WSP_SESSION", "default")
WORK = Path(os.environ.get("WSP_WORK", Path.cwd()))
REFUSED = {"model.language_model.norm.weight", "model.language_model.embed_tokens.weight", "lm_head.weight"}
_st = None


def _safetensors():
    global _st
    if _st is None:
        from huggingface_hub import hf_hub_download

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from wspdelta.safetensors_raw import SafeTensors
        _st = SafeTensors(hf_hub_download("Qwen/Qwen3.5-0.8B", "model.safetensors-00001-of-00001.safetensors",
                                          revision="2fc06364715b967f1860aea9cf38778875588b17"))
    return _st


def _post(path, payload):
    req = urllib.request.Request(SERVER + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=24 * 3600) as resp:
        return json.loads(resp.read())


def names():
    return [n for n in _safetensors().names() if n.startswith("model.language_model.")]


def load_original(name):
    import torch
    from wspdelta import bf16

    return torch.from_numpy(bf16.to_f32(_safetensors().bf16(name)).copy())


def load(name):
    import torch

    path = WORK / "edited" / f"{name}.npy"
    return torch.from_numpy(np.load(path)) if path.exists() else load_original(name)


def save_edit(name, tensor):
    if name in REFUSED:
        raise ValueError(f"{name} may not be edited")
    value = tensor.detach().cpu().float().numpy() if hasattr(tensor, "detach") else np.asarray(tensor, np.float32)
    original = load_original(name).numpy()
    if value.shape != original.shape:
        raise ValueError("shape mismatch")
    ratio = value[original != 0] / original[original != 0]
    if ratio.size and np.allclose(ratio, ratio.flat[0]) and not np.allclose(ratio, 1):
        raise ValueError("an edit that only multiplies a whole tensor by a constant is refused")
    (WORK / "edited").mkdir(parents=True, exist_ok=True)
    path = WORK / "edited" / f"{name}.npy"
    np.save(path, value.astype(np.float32))
    return _post("/edit", {"session": SESSION, "op": "save", "name": name, "npy": str(path)})


def drop_edit(name):
    path = WORK / "edited" / f"{name}.npy"
    if path.exists():
        path.unlink()
    return _post("/edit", {"session": SESSION, "op": "drop", "name": name})


def edited():
    return sorted(p.stem for p in (WORK / "edited").glob("*.npy")) if (WORK / "edited").exists() else []


def forward(request):
    return _post("/forward", {"session": SESSION, "request": request})


if __name__ == "__main__":  # python -m wspserve.client request.json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    print(json.dumps(forward(json.loads(args.request.read_text())), indent=1))
