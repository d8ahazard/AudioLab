import base64
import io
import os
import zlib
from typing import Union

import ffmpeg
import librosa
import numpy as np
import soundfile as sf

MAX_INT16 = 32768
SUPPORTED_AUDIO = ["mp3", "flac", "wav"]  # ogg breaks soundfile
OUTPUT_CHANNELS = ["mono", "stereo"]
AUTOTUNE_NOTES = np.array([
    65.41, 69.30, 73.42, 77.78, 82.41, 87.31,
    92.50, 98.00, 103.83, 110.00, 116.54, 123.47,
    130.81, 138.59, 146.83, 155.56, 164.81, 174.61,
    185.00, 196.00, 207.65, 220.00, 233.08, 246.94,
    261.63, 277.18, 293.66, 311.13, 329.63, 349.23,
    369.99, 392.00, 415.30, 440.00, 466.16, 493.88,
    523.25, 554.37, 587.33, 622.25, 659.25, 698.46,
    739.99, 783.99, 830.61, 880.00, 932.33, 987.77,
    1046.50, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91,
    1479.98, 1567.98, 1661.22, 1760.00, 1864.66, 1975.53,
    2093.00, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83,
    2959.96, 3135.96, 3322.44, 3520.00, 3729.31, 3951.07
])


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


def load_audio(file, sr, **kwargs):
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

    return remix_audio((np.frombuffer(out, np.float32).flatten(), sr), **kwargs)


def remix_audio(input_audio, target_sr=None, norm=False, to_int16=False, resample=False, axis=0, **kwargs):
    audio = np.array(input_audio[0], dtype="float32")
    if target_sr is None: target_sr = input_audio[1]

    print(
        f"before remix: shape={audio.shape}, max={audio.max()}, min={audio.min()}, mean={audio.mean()} sr={input_audio[1]}")
    if resample or input_audio[1] != target_sr:
        audio = librosa.core.resample(np.array(input_audio[0], dtype="float32"), orig_sr=input_audio[1],
                                      target_sr=target_sr, **kwargs)

    if audio.ndim > 1: audio = np.nanmedian(audio, axis=axis)
    if norm: audio = librosa.util.normalize(audio, axis=axis)

    audio_max = np.abs(audio).max() / .99
    if audio_max > 1: audio = audio / audio_max

    if to_int16: audio = np.clip(audio * MAX_INT16, a_min=-MAX_INT16 + 1, a_max=MAX_INT16 - 1).astype("int16")
    print(
        f"after remix: shape={audio.shape}, max={audio.max()}, min={audio.min()}, mean={audio.mean()}, sr={target_sr}")

    return audio, target_sr


def load_input_audio(fname, sr=None, **kwargs):
    if sr is None: sr = 44100
    audio, sr = load_audio(fname, sr, **kwargs)
    # sound = librosa.load(fname,sr=sr,**kwargs)
    print(f"loading sound {fname=} {audio.ndim=} {audio.max()=} {audio.min()=} {audio.dtype=} {sr=}")
    return audio, sr


def save_input_audio(fname, input_audio, sr=None, to_int16=False, to_stereo=False):
    print(f"saving sound to {fname}")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    audio = np.array(input_audio[0], dtype="float32")

    if to_int16:
        audio_max = np.abs(audio).max() / .99
        if audio_max > 1: audio = audio / audio_max
        audio = np.clip(audio * MAX_INT16, a_min=-MAX_INT16 + 1, a_max=MAX_INT16 - 1)

    if to_stereo and audio.ndim < 2: audio = np.stack([audio, audio], axis=-1)
    print(f"{audio.shape=}")

    try:
        sf.write(fname, audio.astype("int16" if np.abs(audio).max() > 1 else "float32"), sr if sr else input_audio[1])
        return f"File saved to ${fname}"
    except Exception as e:
        return f"failed to save audio: {e}"


def audio_to_bytes(audio, sr, format='WAV'):
    bytes_io = io.BytesIO()
    sf.write(bytes_io, audio, sr, format=format)
    return bytes_io.read()


def bytes_to_audio(data: Union[io.BytesIO, bytes], **kwargs):
    if type(data) == bytes:
        bytes_io = io.BytesIO(data)
    else:
        bytes_io = data

    # audio,sr = librosa.load(bytes_io)
    audio, sr = sf.read(bytes_io, **kwargs)
    if audio.ndim > 1:
        if audio.shape[-1] < audio.shape[0]:  # is channel-last format
            audio = audio.T  # transpose to channels-first
    return audio, sr


def bytes2audio(data: str):
    try:
        # Split the suffixed data by the colon
        dtype, data, shape, sr = data.split(":")

        # Get the data, the dtype, and the shape from the split data
        shape = tuple(map(int, shape.split(",")))
        sr = int(sr)

        # Decode the data using base64
        decoded_data = base64.b64decode(data)

        # Decompress the decoded data using zlib
        decompressed_data = zlib.decompress(decoded_data)

        # Convert the decompressed data to a numpy array with the given dtype
        arr = np.frombuffer(decompressed_data, dtype=dtype)

        # Reshape the array to the original shape
        arr = arr.reshape(shape)
        return arr, sr
    except Exception as e:
        print(e)
    return None


def audio2bytes(audio: np.array, sr: int):
    try:
        # Get the dtype, the shape, and the data of the array
        dtype = audio.dtype.name
        shape = audio.shape
        data = audio.tobytes()

        # Compress the data using zlib
        compressed_data = zlib.compress(data)

        # Encode the compressed data using base64
        encoded_data = base64.b64encode(compressed_data)

        # Add a suffix with the dtype and the shape to the encoded data
        suffixed_data = ":".join([dtype, encoded_data.decode(), ",".join(map(str, shape)), str(sr)])
        return suffixed_data
    except Exception as e:
        print(e)
    return ""


def pad_audio(*audios, axis=0):
    maxlen = max(len(a) if a is not None else 0 for a in audios)
    if maxlen > 0:
        stack = librosa.util.stack([librosa.util.pad_center(data=a, size=maxlen) for a in audios if a is not None],
                                   axis=axis)
        return stack
    else:
        return np.stack(audios, axis=axis)


def merge_audio(audio1, audio2, sr=40000):
    print(f"merging audio audio1={audio1[0].shape, audio1[1]} audio2={audio2[0].shape, audio2[1]} sr={sr}")
    m1, _ = remix_audio(audio1, target_sr=sr, axis=0)
    m2, _ = remix_audio(audio2, target_sr=sr, axis=0)

    mixed = pad_audio(m1, m2, axis=0)

    return remix_audio((mixed, sr), to_int16=True, axis=0, norm=True)


def autotune_f0(f0, threshold=0.):
    print("autotuning f0 using note_dict...")

    autotuned_f0 = []
    # Loop through each value in array1
    for freq in f0:
        # Find the absolute difference between x and each value in array2
        diff = np.abs(AUTOTUNE_NOTES - freq)
        # Find the index of the minimum difference
        idx = np.argmin(diff)
        # Find the corresponding value in array2
        y = AUTOTUNE_NOTES[idx]
        # Check if the difference is less than threshold
        if diff[idx] < threshold:
            # Keep the value in array1
            autotuned_f0.append(freq)
        else:
            # Use the nearest value in array2
            autotuned_f0.append(y)
    # Return the result as a numpy array
    return np.array(autotuned_f0, dtype="float32")


def remap_f0(
    f0,
    octave_range=2,
    mad_multiplier=2.0,
    local_window=3,
    local_octave_margin=0.25
):
    """
    Remap octave outliers in f0 by:
      1) Detecting global outliers in log2 space (median + MAD).
      2) For each outlier, looking at a small local neighborhood (±local_window).
         We find the local pitch range (in log2 space) and see if shifting by ±1..±octave_range
         can bring the note into that local range (±local_octave_margin around the local median).
      3) If local shifting doesn't fix it, fall back to a global shift that brings it closer
         to the global median. Otherwise, leave as-is.

    Parameters:
      f0 (np.array): Pitch array (floats). 0 or negative means unvoiced.
      octave_range (int): How many octaves up/down to attempt for outlier correction.
      mad_multiplier (float): Outlier threshold multiplier for the MAD in log2 space.
      local_window (int): How many frames on each side to consider for local context.
      local_octave_margin (float): Local range half-width in log2 space, e.g. 0.25 is ±1/4 octave.

    Returns:
      np.array: The remapped pitch array.
    """
    f0 = np.array(f0, dtype=np.float32)

    # Identify global median + MAD in log2 space for outlier detection
    nonzero = f0[f0 > 0]
    if len(nonzero) == 0:
        return f0  # Nothing to do

    log2_nonzero = np.log2(nonzero)
    median_log2 = np.median(log2_nonzero)
    abs_devs = np.abs(log2_nonzero - median_log2)
    mad_log2 = np.median(abs_devs) if len(abs_devs) > 0 else 0.0

    # If there's effectively no variation, just return
    if mad_log2 < 1e-9:
        return f0

    outlier_thresh = mad_multiplier * mad_log2

    # We'll store results here
    remapped = []
    n_frames = len(f0)

    for i, freq in enumerate(f0):
        # Unvoiced or invalid
        if freq <= 0:
            remapped.append(freq)
            continue

        freq_log2 = np.log2(freq)
        diff_global = abs(freq_log2 - median_log2)

        # If not a global outlier, leave as-is
        if diff_global <= outlier_thresh:
            remapped.append(freq)
            continue

        # Otherwise, we have a global outlier; attempt local fix
        start_idx = max(0, i - local_window)
        end_idx = min(n_frames, i + local_window + 1)
        local_region = f0[start_idx:end_idx]
        local_nonzero = local_region[local_region > 0]

        # If local region has no voiced frames, just do a global fallback
        if len(local_nonzero) == 0:
            best_candidate = freq
            best_diff = diff_global
            # Try shifting globally
            for shift in range(-octave_range, octave_range + 1):
                if shift == 0:
                    continue
                candidate_log2 = freq_log2 + shift
                candidate_diff = abs(candidate_log2 - median_log2)
                if candidate_diff < best_diff:
                    best_diff = candidate_diff
                    best_candidate = 2 ** candidate_log2
            # If that global shift is no longer an outlier, adopt it
            if best_diff <= outlier_thresh:
                remapped.append(best_candidate)
            else:
                remapped.append(freq)
            continue

        # We have some local pitched frames
        local_log2 = np.log2(local_nonzero)
        local_median_log2 = np.median(local_log2)
        # Define local range around that median
        local_lower = local_median_log2 - local_octave_margin
        local_upper = local_median_log2 + local_octave_margin

        # 1) Try local shifting first
        best_candidate = freq
        best_diff_local = float("inf")

        for shift in range(-octave_range, octave_range + 1):
            if shift == 0:
                continue
            candidate_log2 = freq_log2 + shift
            # If candidate is within the local "reasonable" range, consider it
            if local_lower <= candidate_log2 <= local_upper:
                # measure difference from local median
                candidate_diff = abs(candidate_log2 - local_median_log2)
                if candidate_diff < best_diff_local:
                    best_diff_local = candidate_diff
                    best_candidate = 2 ** candidate_log2

        # If we found a local shift that puts freq in the local range
        if best_diff_local < float("inf"):
            remapped.append(best_candidate)
            continue

        # 2) If no local shift fixed it, do the old global approach
        best_candidate = freq
        best_diff_global = diff_global

        for shift in range(-octave_range, octave_range + 1):
            if shift == 0:
                continue
            candidate_log2 = freq_log2 + shift
            candidate_diff = abs(candidate_log2 - median_log2)
            if candidate_diff < best_diff_global:
                best_diff_global = candidate_diff
                best_candidate = 2 ** candidate_log2

        # If that global shift is no longer an outlier, adopt it
        if best_diff_global <= outlier_thresh:
            remapped.append(best_candidate)
        else:
            # Keep original if neither local nor global shift helps
            remapped.append(freq)

    return np.array(remapped, dtype=np.float32)

