"""Compatibility entry point; AudioLab now configures each separator instance.

AudioSeparator 0.47.0 owns ONNX loading and CPU/CUDA provider selection.
The old process-wide patch preferred TensorRT and swallowed load failures.
"""


def patch_separator():
    """Retained for callers importing the old helper; no global patch is applied."""
    return None
