import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)


def clamp_tempo_factor(tempo_factor: float, min_factor: float = 0.5, max_factor: float = 1.5) -> float:
    return max(min_factor, min(max_factor, tempo_factor))


def adjust_tempo(input_path: str, output_path: str, tempo_factor: float) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tempo_factor = clamp_tempo_factor(tempo_factor)

    if abs(tempo_factor - 1.0) < 1e-6:
        shutil.copyfile(input_path, output_path)
        return output_path

    filter_str = f"rubberband=tempo={tempo_factor}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-filter:a",
        filter_str,
        "-acodec",
        "pcm_s16le",
        output_path,
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logger.error(f"Tempo adjustment failed: {result.stderr}")
        raise RuntimeError(f"Tempo adjustment failed: {result.stderr}")

    return output_path
