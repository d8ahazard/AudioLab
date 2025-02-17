import av
import ffmpeg
import librosa
import numpy as np
import soundfile as sf


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


def load_audio_advanced(file, sr=16000, mono=False, return_sr=False):
    """
    Loads an audio file with optional resampling to 'sr' and optional downmixing to mono.
    Returns a NumPy array of shape:
      - (num_samples,) if mono or single-channel
      - (num_samples, num_channels) if stereo (or multi-channel) and mono=False
    """
    # Read file with soundfile
    audio, sr_in = sf.read(file)  # shape is (num_samples, num_channels) if stereo

    # If needed, resample using librosa
    if sr_in != sr and sr:
        # Use named arguments to avoid version-specific issues
        audio = librosa.resample(y=audio.T, orig_sr=sr_in, target_sr=sr).T
        # Now 'audio' is shape (num_samples, num_channels)

    # Optionally downmix to mono
    if mono and audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio if not return_sr else (audio, sr_in)


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()