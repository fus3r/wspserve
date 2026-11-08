"""Exact bfloat16 bit handling with NumPy (NumPy has no bfloat16 dtype)."""
import hashlib

import numpy as np

MAX_FINITE_MAGNITUDE = 0x7F7F


def to_f32(bits: np.ndarray) -> np.ndarray:
    """bfloat16 bits (uint16) to float32 values, exact."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def to_f64(bits: np.ndarray) -> np.ndarray:
    return to_f32(bits).astype(np.float64)


def from_f32(values: np.ndarray) -> np.ndarray:
    """float32 values to bfloat16 bits, rounding to nearest even (PyTorch's conversion)."""
    x = np.ascontiguousarray(values, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    out = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(x)
    if nan.any():
        out[nan] = ((u[nan] >> 16) | 0x40).astype(np.uint16)
    return out


def from_f64(values: np.ndarray) -> np.ndarray:
    """float64 -> float32 -> bfloat16, the double rounding used by the authors' rebuild code."""
    return from_f32(np.asarray(values, dtype=np.float64).astype(np.float32))


def sha256(bits: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(bits, dtype=np.uint16).tobytes()).hexdigest()


def rounding_interval(bits: np.ndarray, shrink: float = 0.49):
    """Float64 bounds (slightly shrunk) of the values that round to these bfloat16 bits.

    Used only to steer a factor search, every candidate is verified bit for bit afterwards.
    """
    bits = np.asarray(bits, dtype=np.uint16)
    magnitude = (bits & 0x7FFF).astype(np.int64)

    def value(m):
        return to_f64(np.clip(m, 0, MAX_FINITE_MAGNITUDE).astype(np.uint16))

    center = value(magnitude)
    up = value(magnitude + 1) - center
    down = np.where(magnitude == 0, 0.0, center - value(magnitude - 1))
    low, high = center - shrink * down, center + shrink * up
    negative = (bits & 0x8000) != 0
    return np.where(negative, -high, low), np.where(negative, -low, high)
