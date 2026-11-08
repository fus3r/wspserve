"""wspdelta: compact, verifiable representations of weight-space-programming edits.

- ``structure.analyze`` recovers a bit-exact typed representation of a BF16 edit,
- ``delta`` defines the types (ChannelScale, LowRank, SparseOverlay),
- ``runtime.MultiVersion`` serves several versions over one resident base in one batch,
- ``export.peft_adapter`` writes an edit as a PEFT LoRA adapter.

NumPy-only except ``runtime`` and ``export``, which need PyTorch.
"""
from .delta import ChannelScale, LowRank, SparseOverlay, serving_cost
from .structure import analyze, changed, fit_channel_scale, fit_low_rank

__all__ = ["ChannelScale", "LowRank", "SparseOverlay", "analyze", "changed", "fit_channel_scale",
           "fit_low_rank", "serving_cost"]
