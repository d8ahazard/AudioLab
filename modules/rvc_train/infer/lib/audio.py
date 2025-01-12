import os
import traceback

import librosa
import numpy as np
import av
from io import BytesIO


def wav2(i, o, format):
    inp = av.open(i, "rb")
    if format == "m4a":
        format = "mp4"
    out = av.open(o, "wb", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def audio2(input_file, output_file, format, sr):
    """
    Convert audio file into another format
    :param format: output file format
    :param sr: sample rate
    :return: None
    """
    inp = av.open(input_file, "r")
    out = av.open(output_file, "w", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "f32le":
        format = "pcm_f32le"

    ostream = out.add_stream(format, channels=1)
    ostream.sample_rate = sr

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    out.close()
    inp.close()


def load_audio(file, sr):
    """
    Load audio file into numpy array
    format: pcm_f32le which means float32 little-endian
    :param file: path to audio file
    :param sr: sample rate
    :return: numpy array

    >>> load_audio("tests/data/1.wav", 16000)
    array([ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00, ..., -1.5258789e-05,
       -1.5258789e-05, -1.5258789e-05], dtype=float32)
    """
    file = (
        str(file).strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    )
    if os.path.exists(file) == False:
        raise RuntimeError(
            "You input a wrong audio path that does not exists, please fix it!"
        )
    try:
        with open(file, "rb") as f:
            with BytesIO() as out:
                audio2(f, out, "f32le", sr)
                return np.frombuffer(out.getvalue(), np.float32).flatten()

    except AttributeError:
        audio = file[1] / 32768.0
        if len(audio.shape) == 2:
            audio = np.mean(audio, -1)
        return librosa.resample(audio, orig_sr=file[0], target_sr=16000)

    except:
        raise RuntimeError(traceback.format_exc())
