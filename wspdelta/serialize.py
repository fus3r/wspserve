"""Save and load compiled edit representations as a single .npz (no pickle)."""
import json

import numpy as np

from .delta import ChannelScale, LowRank, SparseOverlay


def save(path, deltas):
    arrays, index = {}, []
    for i, (name, delta) in enumerate(deltas.items()):
        entry = {"name": name, "kind": delta.kind, "key": f"d{i}"}
        if isinstance(delta, ChannelScale):
            arrays.update({f"d{i}_columns": delta.columns, f"d{i}_scales": delta.scales,
                           f"d{i}_assigned_zero": delta.assigned_zero})
        elif isinstance(delta, LowRank):
            support = delta.support
            arrays.update({f"d{i}_b": delta.b, f"d{i}_a": delta.a[:, support], f"d{i}_support": support,
                           f"d{i}_fixup_index": delta.fixup_index, f"d{i}_fixup_bits": delta.fixup_bits})
            entry["inputs"] = int(delta.a.shape[1])
        elif isinstance(delta, SparseOverlay):
            arrays.update({f"d{i}_index": delta.index, f"d{i}_bits": delta.bits})
        else:
            raise TypeError(type(delta))
        index.append(entry)
    arrays["index"] = np.frombuffer(json.dumps(index).encode(), dtype=np.uint8)
    np.savez_compressed(path, **arrays)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        out = {}
        for entry in json.loads(bytes(z["index"]).decode()):
            k = entry["key"]
            if entry["kind"] == "channel_scale":
                out[entry["name"]] = ChannelScale(z[f"{k}_columns"], z[f"{k}_scales"], z[f"{k}_assigned_zero"])
            elif entry["kind"] == "low_rank":
                a = np.zeros((z[f"{k}_a"].shape[0], entry["inputs"]))
                a[:, z[f"{k}_support"]] = z[f"{k}_a"]
                out[entry["name"]] = LowRank(z[f"{k}_b"], a, z[f"{k}_fixup_index"], z[f"{k}_fixup_bits"])
            else:
                out[entry["name"]] = SparseOverlay(z[f"{k}_index"], z[f"{k}_bits"])
        return out
