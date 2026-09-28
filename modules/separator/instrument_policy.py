"""Listener-selected instruments; Mega adds only uncovered leaf categories."""
import numpy as np

INSTRUMENT_STEMS = ['drums', 'bass', 'guitar', 'piano', 'other']
DRUM_STEMS = ['kick', 'snare', 'hh', 'toms', 'ride', 'crash']
MEGA_MODEL = 'mvsep_mega_model_bs_roformer_53_stems_v1.ckpt'
MEGA_CONFIG = 'mvsep_mega_model_bs_roformer_53_stems.yaml'
# Exclude vocals, main instruments, kit components and overlapping family heads.
# Keep specific percussion that DrumSep does not predict.
MEGA_EXTRAS = ['accordion', 'banjo', 'bassoon', 'cello', 'clarinet', 'congas',
    'flute', 'french-horn', 'glockenspiel', 'harmonica', 'harp', 'harpsichord', 'mandolin',
    'marimba', 'oboe', 'organ', 'saxophone', 'sitar', 'synth', 'tambourine', 'timpani',
    'triangle', 'trombone', 'trumpet', 'tuba', 'ukulele', 'viola', 'violin', 'wind-chimes']


def selected(options, key, allowed, default):
    value = options.get(key, default)
    if not isinstance(value, (list, tuple)) or any(v not in allowed for v in value):
        raise ValueError(f'{key} must be a list chosen from {allowed}')
    return list(dict.fromkeys(value))


def duplicate_of(audio, candidates):
    """Flag near-identical same-gain copies, not merely correlated instruments.

    No gain fitting or temporal shifts: unison instruments must not be merged.
    Very quiet/empty estimates are not considered duplicate evidence.
    """
    a = np.asarray(audio, dtype=np.float32)
    probe = a[:, ::64]
    power = np.mean(probe*probe, dtype=np.float64)
    if power < 1e-12:
        return None
    for role, other in candidates.items():
        b = np.asarray(other, dtype=np.float32)
        if b.shape != a.shape or np.mean((probe-b[:, ::64])**2, dtype=np.float64) / power >= 1e-8:
            continue
        full_power = np.mean(a*a, dtype=np.float64)
        if full_power > 1e-12 and np.mean((a-b)**2, dtype=np.float64) / full_power < 1e-8:
            return role
    return None
