"""Read raw tensors from a .safetensors file without a framework, so BF16 bits stay exact."""
import json
import struct

import numpy as np

_DTYPES = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "F64": np.float64,
           "I64": np.int64, "I32": np.int32, "U8": np.uint8, "BOOL": np.bool_}


class SafeTensors:
    def __init__(self, path):
        self.path = str(path)
        with open(path, "rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            self.header = json.loads(handle.read(length))
        self.data_start = 8 + length
        self._map = np.memmap(self.path, dtype=np.uint8, mode="r")

    def names(self):
        return [name for name in self.header if name != "__metadata__"]

    def info(self, name):
        entry = self.header[name]
        return entry["dtype"], list(entry["shape"])

    def raw(self, name) -> np.ndarray:
        """The tensor as stored (BF16 as uint16 bit patterns), copied out of the map."""
        entry = self.header[name]
        start, end = entry["data_offsets"]
        dtype = _DTYPES[entry["dtype"]]
        flat = self._map[self.data_start + start:self.data_start + end].view(dtype)
        return np.array(flat).reshape(entry["shape"])

    def bf16(self, name) -> np.ndarray:
        dtype, _ = self.info(name)
        if dtype != "BF16":
            raise ValueError(f"{name} is {dtype}, not BF16")
        return self.raw(name)
