import base64
import io
import math
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

    if resample or input_audio[1] != target_sr:
        audio = librosa.core.resample(np.array(input_audio[0], dtype="float32"), orig_sr=input_audio[1],
                                      target_sr=target_sr, **kwargs)

    if audio.ndim > 1: audio = np.nanmedian(audio, axis=axis)
    if norm: audio = librosa.util.normalize(audio, axis=axis)

    audio_max = np.abs(audio).max() / .99
    if audio_max > 1: audio = audio / audio_max

    if to_int16: audio = np.clip(audio * MAX_INT16, a_min=-MAX_INT16 + 1, a_max=MAX_INT16 - 1).astype("int16")

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


def _compute_local_medians(log2_array, window_size):
    """
    Compute a local median in log2 space for each frame, ignoring NaNs.
    Naive O(N*window_size) implementation for clarity.
    """
    n = len(log2_array)
    local_medians = np.full_like(log2_array, np.nan, dtype=np.float32)

    for i in range(n):
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        window_vals = log2_array[start:end]
        # Filter out NaNs
        window_vals = window_vals[~np.isnan(window_vals)]
        if len(window_vals) > 0:
            local_medians[i] = np.median(window_vals)
        else:
            # If no valid frames, just leave NaN
            local_medians[i] = np.nan

    return local_medians


def remap_f0(
        f0,
        window_size=4,
        outlier_thresh=0.5,
        octave_range=1
):
    """
    Remap octave outliers using local context:
      1) Convert f0 to log2 space, ignoring unvoiced frames (f0 <= 0).
      2) Compute a local median of log2 pitches for each frame in a +/- window_size neighborhood.
      3) Any frame that differs from its local median by > outlier_thresh (in log2) is considered an outlier.
      4) For outliers, try shifting by ±1..±octave_range octaves to see if it lands within outlier_thresh
         of the local median. If yes, adopt that shift. Otherwise, leave it.

    Parameters:
      f0 (np.array): Array of pitches in Hz. 0 or negative => unvoiced.
      window_size (int): Radius of the local neighborhood to compute the median.
      outlier_thresh (float): How far (in log2 space) from the local median is considered outlier.
                              e.g. 0.5 => ±0.5 octaves from local median.
      octave_range (int): Max number of octaves up/down to try for outlier correction.

    Returns:
      np.array: Remapped pitch array of the same shape.
    """
    f0 = np.array(f0, dtype=np.float32)
    n = len(f0)

    # Convert to log2, store NaN for unvoiced
    log2_f0 = np.full_like(f0, np.nan, dtype=np.float32)
    voiced_mask = (f0 > 0)
    log2_f0[voiced_mask] = np.log2(f0[voiced_mask])

    # Compute local medians ignoring NaNs
    local_medians = _compute_local_medians(log2_f0, window_size)

    # Output array
    remapped = np.copy(f0)

    for i in range(n):
        if not voiced_mask[i]:
            # Unvoiced => skip
            continue

        freq_log2 = log2_f0[i]
        local_med = local_medians[i]

        # If local median is NaN (no neighbors?), skip
        if math.isnan(local_med):
            continue

        diff = abs(freq_log2 - local_med)
        if diff <= outlier_thresh:
            # Not an outlier
            continue

        # It's an outlier => try shifting
        best_candidate = freq_log2
        best_diff = diff

        for shift in range(-octave_range, octave_range + 1):
            if shift == 0:
                continue
            candidate_log2 = freq_log2 + shift
            candidate_diff = abs(candidate_log2 - local_med)
            if candidate_diff < best_diff:
                best_diff = candidate_diff
                best_candidate = candidate_log2

        # If the best shift gets us within outlier_thresh, adopt it
        if best_diff <= outlier_thresh:
            remapped[i] = 2 ** best_candidate
        # else keep original

    return remapped
