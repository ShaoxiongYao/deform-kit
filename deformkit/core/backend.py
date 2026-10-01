"""Device selection and small NumPy <-> Warp helpers."""

import numpy as np
import warp as wp

_initialized = False


def init():
    global _initialized
    if not _initialized:
        wp.config.quiet = True
        wp.init()
        _initialized = True


def get_device(device=None):
    """Return a Warp device, defaulting to the first GPU when one is available."""
    init()
    if device is not None:
        return wp.get_device(device)
    return wp.get_device("cuda:0" if wp.get_cuda_device_count() > 0 else "cpu")


def to_vec3(a, device):
    return wp.array(np.asarray(a, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3, device=device)


def to_float(a, device):
    return wp.array(np.asarray(a, dtype=np.float32).reshape(-1), dtype=wp.float32, device=device)


def to_int(a, device, dtype=wp.int32):
    return wp.array(np.asarray(a, dtype=np.int32), dtype=dtype, device=device)
