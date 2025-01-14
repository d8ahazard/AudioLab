import os
import traceback

import librosa
import numpy as np
import av
from io import BytesIO


def wav2(input_file, output_file, container_fmt, sr=44100):
    """
    Convert audio input_file -> output_file with a certain container_fmt (e.g., 'wav', 'ogg'),
    forcing sample rate (sr) and single-channel (mono).
    """
    if container_fmt == "ogg":
        codec_name = "libvorbis"
    elif container_fmt == "f32le":
        # 'f32le' is a PCM codec. If you want an actual WAV container, set container_fmt = 'wav' instead.
        codec_name = "pcm_f32le"
    else:
        codec_name = container_fmt

    inp = av.open(input_file, mode="r")
    out = av.open(output_file, mode="w", format=container_fmt)

    # Put the codec name as the FIRST positional argument:
    # (PyAV often needs at least one positional arg, i.e. the codec name.)
    #
    # Then pass sample rate and layout as keyword args.
    # Some older builds of PyAV also choke on 'layout="mono"'—
    # if that happens, omit layout and only pass `rate=sr`.
    ostream = out.add_stream(codec_name, rate=sr, layout="mono")

    for frame in inp.decode(audio=0):
        for packet in ostream.encode(frame):
            out.mux(packet)

    # Flush any buffered packets
    for packet in ostream.encode(None):
        out.mux(packet)

    out.close()
    inp.close()


def load_audio(file_path, sr=16000):
    """
    Load an audio file into a float32 NumPy array at the given sample rate (sr).
    First try PyAV in-memory decoding to float32 (mono).
    Fallback to librosa if PyAV fails.
    """
    file_path = str(file_path).strip('"').strip()

    if not os.path.exists(file_path):
        raise RuntimeError(f"Audio path not found: {file_path}")

    try:
        # Decode to PCM float32 in memory
        with open(file_path, "rb") as f, BytesIO() as out_buf:
            wav2(f, out_buf, "f32le", sr=sr)  # decode -> float32, 1-channel
            return np.frombuffer(out_buf.getvalue(), dtype=np.float32).flatten()

    except AttributeError:
        # If PyAV has issues, fallback to librosa
        print(f"PyAV decoding failed for {file_path}, falling back to librosa...")
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        return audio.astype(np.float32)

    except Exception:
        raise RuntimeError(
            f"Could not load audio from {file_path}\n{traceback.format_exc()}"
        )
