import numpy as np


def stereo_to_mono_ms(stereo_track: np.ndarray):
    left = stereo_track[:, 0]
    right = stereo_track[:, 1]
    mono = 0.5 * (left + right)
    side = 0.5 * (left - right)
    return mono.astype(np.float32), side.astype(np.float32), len(mono)


def resample_side(side: np.ndarray, orig_len: int, new_len: int):
    if new_len == orig_len:
        return side
    old_indices = np.linspace(0, orig_len - 1, orig_len)
    new_indices = np.linspace(0, orig_len - 1, new_len)
    return np.interp(new_indices, old_indices, side).astype(np.float32)


def mono_to_stereo_ms(mono: np.ndarray, side: np.ndarray):
    left = mono + side
    right = mono - side
    return np.column_stack([left, right]).astype(np.float32)
