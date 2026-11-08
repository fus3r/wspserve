"""Write compiled edits as a PEFT LoRA adapter (the format multi-LoRA servers load)."""
import json
from pathlib import Path

import numpy as np


def peft_adapter(deltas, base_bits, out_dir, *, base_model, revision, module_prefix=""):
    """One adapter per edit version. Ranks are padded with zero factors to the largest rank.

    W' - W = B A exactly in real arithmetic, so lora_alpha = r makes the PEFT scale 1.
    Factors are stored in float32, a server may cast them to its compute dtype.
    """
    import torch
    from safetensors.torch import save_file

    factors = {name: delta.lora(base_bits[name]) for name, delta in deltas.items()}
    rank = max(a.shape[0] for a, _ in factors.values())
    tensors, targets = {}, set()
    for name, (a, b) in factors.items():
        stem = module_prefix + name.removesuffix(".weight")
        pad = rank - a.shape[0]
        a = np.vstack([a, np.zeros((pad, a.shape[1]))]) if pad else a
        b = np.hstack([b, np.zeros((b.shape[0], pad))]) if pad else b
        tensors[f"base_model.model.{stem}.lora_A.weight"] = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        tensors[f"base_model.model.{stem}.lora_B.weight"] = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32))
        targets.add(stem.rsplit(".", 1)[-1])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "adapter_model.safetensors"))
    config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base_model,
              "revision": revision, "r": int(rank), "lora_alpha": int(rank), "lora_dropout": 0.0,
              "bias": "none", "fan_in_fan_out": False, "use_rslora": False,
              "target_modules": sorted(targets),
              "layers_to_transform": None, "modules_to_save": None}
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return {"rank": int(rank), "target_modules": sorted(targets), "modules": len(factors),
            "bytes": int((out_dir / "adapter_model.safetensors").stat().st_size)}
