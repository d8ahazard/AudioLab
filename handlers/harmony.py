import numpy as np
import librosa
import logging
import soundfile as sf

logger = logging.getLogger(__name__)


def extract_pitch(audio, sr):
    """
    Extracts the pitch contour from an audio signal using librosa's pyin.
    Returns an array of pitch frequencies (Hz) where unvoiced frames are NaN.
    """
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )
    return f0


def detect_chord_notes(pitch_contour, sr, hop_length, window_sec=1.0):
    """
    Segments the pitch contour into fixed-length windows and returns
    a list of chord note names (one per segment). If no pitch is detected
    in a window, None is returned for that segment.
    """
    # Convert window size from seconds to number of frames
    frames_per_sec = sr / hop_length
    window_size = int(window_sec * frames_per_sec)
    chord_notes = []

    for i in range(0, len(pitch_contour), window_size):
        window = pitch_contour[i:i + window_size]
        # Remove NaNs (unvoiced frames)
        valid_pitches = window[~np.isnan(window)]
        if len(valid_pitches) == 0:
            chord_notes.append(None)
        else:
            # Use median pitch as representative for the window
            median_pitch = np.median(valid_pitches)
            note = librosa.hz_to_note(median_pitch)
            chord_notes.append(note)

    return chord_notes


def pitch_shift_segment(segment, sr, shift_semitones):
    """
    Pitch shifts a segment of audio by the specified semitones.
    """
    return librosa.effects.pitch_shift(segment, sr=sr, n_steps=shift_semitones)


def recreate_harmonies(background_path, main_vocal_path, output_path, hop_length=512, window_sec=1.0):
    """
    Analyzes the background vocal track to extract chord notes, and then
    uses the main vocal track to re-create these harmonies by pitch shifting
    segments of the main vocal according to the detected chord.

    Parameters:
    - background_path: path to the background vocal track (with layered harmonies)
    - main_vocal_path: path to the main vocal track
    - output_path: where to save the harmonized track
    - hop_length: number of samples between successive frames for analysis
    - window_sec: duration (in seconds) of each segment/window for chord detection

    Returns:
    - output_path (str): the path to the saved harmonized audio file.
    """
    # Load background and main vocal tracks (preserving original sampling rate)
    bg_audio, sr = librosa.load(background_path, sr=None)
    main_audio, _ = librosa.load(main_vocal_path, sr=sr)

    # Extract pitch contour from background track
    pitch_contour = extract_pitch(bg_audio, sr)

    # Detect chord notes from the pitch contour
    chord_notes = detect_chord_notes(pitch_contour, sr, hop_length, window_sec)

    # Calculate segment length in samples from window_sec
    segment_samples = int(window_sec * sr)
    output_audio = np.zeros_like(main_audio)

    # Define a default reference note (assume main vocal is around C4)
    ref_note = 'C4'
    ref_hz = librosa.note_to_hz(ref_note)

    num_segments = int(np.ceil(len(main_audio) / segment_samples))

    for i in range(num_segments):
        start = i * segment_samples
        end = min(start + segment_samples, len(main_audio))
        segment = main_audio[start:end]

        chord = chord_notes[i] if i < len(chord_notes) else None

        if chord is None:
            # If no chord detected, leave the segment unchanged
            shifted_segment = segment
        else:
            # Calculate semitone shift: log2 ratio * 12
            target_hz = librosa.note_to_hz(chord)
            shift_semitones = np.log2(target_hz / ref_hz) * 12
            shifted_segment = pitch_shift_segment(segment, sr, shift_semitones)

        # Place the (possibly) shifted segment into the output array
        output_audio[start:end] = shifted_segment

    # Write the output to file
    sf.write(output_path, output_audio, sr)
    return output_path
