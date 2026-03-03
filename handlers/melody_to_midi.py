import logging
import os

from basic_pitch.inference import predict

logger = logging.getLogger(__name__)


def extract_melody_to_midi(
    input_path: str,
    output_path: str,
    bpm: int = 120,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length_ms: float = 127.7,
) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"MelodyToMidi: running Basic Pitch on {input_path}")
    _, midi_data, note_events = predict(
        input_path,
        midi_tempo=bpm,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length_ms,
    )
    if not note_events:
        logger.warning("No melody detected, writing empty MIDI.")
    else:
        logger.info(f"MelodyToMidi: extracted {len(note_events)} notes")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    midi_data.write(output_path)
    logger.info(f"MelodyToMidi: saved MIDI {output_path}")
    return output_path
