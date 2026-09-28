"""Causal stereo effect capture with held-out validation and explicit wet-only IRs."""
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import rfft, irfft, next_fast_len
from scipy.signal import fftconvolve, resample_poly
from scipy.sparse.linalg import LinearOperator, cg

REVISION = 'stereo-causal-ridge-1'


class InvalidImpulseResponse(ValueError):
    pass


def capture_for_stem(stem_path, folder):
    """Resolve only the IR owned by this source stem, never a stale shared file."""
    from modules.separator.stem_manifest import read_manifest, safe_path, file_hash
    name = Path(stem_path).stem
    if '(Re-Reverb)' in name:
        return None
    for stem in read_manifest(folder).get('stems', []):
        base = Path(stem['filename']).stem
        if stem.get('role') != 'vocals' or not (name == base or name.startswith(base + '(')):
            continue
        capture = stem.get('reverb_ir')
        if not capture:
            return None
        path = safe_path(folder, capture['path'])
        if file_hash(path) != capture['sha256']:
            raise InvalidImpulseResponse('Capture checksum changed; recapture before restoration')
        params = json.loads(Path(path).read_text(encoding='utf-8'))
        if params.get('exported_wet') and '(Cloned)' not in name:
            return None  # Wet export already contains the captured effect.
        return str(path)
    return None


def audio_array(audio):
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or audio.shape[1] not in (1, 2) or not np.isfinite(audio).all():
        raise ValueError('Expected finite, sample-major mono/stereo audio')
    return audio


def convolve_effect(dry, impulse, keep_tail=False):
    dry, impulse = audio_array(dry), audio_array(impulse)
    channels = max(dry.shape[1], impulse.shape[1])
    if dry.shape[1] == 1:
        dry = np.repeat(dry, channels, axis=1)
    if impulse.shape[1] == 1:
        impulse = np.repeat(impulse, channels, axis=1)
    wet = np.column_stack([fftconvolve(dry[:, c], impulse[:, c]) for c in range(channels)])
    return wet if keep_tail else wet[:len(dry)]


def fit_effect_ir(dry, wet_effect, sr, max_ir_seconds=1.5, regularization=.001, max_iterations=40, segment_frames=None):
    """Fit dry -> removed effect, not dry -> dry+effect. Never fit on holdout.

    Two independent causal channel filters preserve stereo; cross-channel routing
    is deliberately not estimated from highly correlated vocal stereo channels.
    Start-of-block context is excluded from the loss, not treated as zero history.
    """
    dry, wet = audio_array(dry), audio_array(wet_effect)
    if dry.shape != wet.shape or sr <= 0:
        raise ValueError('Capture requires aligned dry/effect arrays of identical shape/rate')
    if not .001 <= max_ir_seconds <= 4 or not 0 < regularization <= 1:
        raise ValueError('Invalid IR length or regularization')
    taps = max(2, round(sr * max_ir_seconds))
    block = segment_frames or min(round(sr * 8), len(dry) // 3)
    if block * 3 > len(dry) or block <= 2 * taps:
        raise ValueError('Need three disjoint blocks, each longer than twice the IR length')
    starts = [0, (len(dry)-block)//2, len(dry)-block]
    blocks = [(dry[s:s+block], wet[s:s+block]) for s in starts]
    nfft = next_fast_len(block + taps - 1)
    mask = np.ones(block)
    mask[:taps] = 0
    filters, solver_status = [], []
    for channel in range(dry.shape[1]):
        train = [(x[:, channel], y[:, channel]) for x, y in blocks[:2]]
        transforms = [rfft(x, nfft) for x, _ in train]
        energy = sum(float(np.dot(x, x)) for x, _ in train)
        if energy < 1e-12:
            filters.append(np.zeros(taps))
            solver_status.append(-1)
            continue
        ridge = regularization * energy
        def adjoint(spectrum, vector):
            return irfft(np.conj(spectrum) * rfft(vector * mask, nfft), nfft)[:taps]
        rhs = sum(adjoint(spectrum, y) for spectrum, (_, y) in zip(transforms, train))
        def normal(h):
            H = rfft(h, nfft)
            return sum(adjoint(X, irfft(X * H, nfft)[:block]) for X in transforms) + ridge * h
        fitted, status = cg(LinearOperator((taps, taps), matvec=normal, dtype=np.float64), rhs,
                            rtol=1e-5, atol=1e-10, maxiter=max_iterations)
        filters.append(fitted)
        solver_status.append(int(status))
    impulse = np.column_stack(filters)
    # A short end taper avoids an abrupt truncated tail; evaluate the actual saved IR.
    taper = min(taps // 10, round(sr * .05))
    if taper:
        impulse[-taper:] *= np.linspace(1, 0, taper)[:, None]
    hold_dry, hold_wet = blocks[2]
    prediction = convolve_effect(hold_dry, impulse)
    target = hold_wet[taps:]
    error = prediction[taps:] - target
    target_energy = float(np.sum(target**2))
    nmse = float(np.sum(error**2) / max(target_energy, 1e-20))
    channel_nmse = np.sum(error**2, axis=0) / np.maximum(np.sum(target**2, axis=0), 1e-20)
    gain = float(np.max(abs(rfft(impulse, next_fast_len(taps*2), axis=0))))
    effect_rms = float(np.sqrt(np.mean(wet**2)))
    dry_rms = float(np.sqrt(np.mean(dry**2)))
    active_channels = np.sum(target**2, axis=0) > max(1e-14, target_energy * .01)
    no_effect = effect_rms < 1e-8
    reasons = []
    if dry_rms < 1e-7:
        reasons.append('insufficient dry excitation')
    if not no_effect and (target_energy < 1e-12 or nmse > .9):
        reasons.append('less than 10% held-out effect error reduction')
    if not no_effect and np.any(channel_nmse[active_channels] > 1.):
        reasons.append('held-out channel worse than applying no effect')
    if gain > 8 or not np.isfinite(impulse).all():
        reasons.append('unstable transfer gain')
    if any(status < 0 for status in solver_status):
        reasons.append('unexcited channel or solver failure')
    return {'schema': 2, 'revision': REVISION, 'sample_rate': sr, 'kind': 'wet_only_diagonal_stereo',
            'impulse_response': impulse.tolist(), 'valid_for_restore': not reasons,
            'rejection_reasons': reasons, 'settings': {'max_ir_seconds': max_ir_seconds,
                'regularization': regularization, 'max_iterations': max_iterations},
            'diagnostics': {'holdout_effect_nmse': nmse, 'holdout_channel_nmse': channel_nmse.tolist(),
                'holdout_effect_error_reduction_percent': 100 * (1-nmse),
                'maximum_frequency_gain': gain, 'solver_status': solver_status,
                'effect_rms': effect_rms, 'dry_rms': dry_rms,
                'train_spans_frames': [[s, s+block] for s in starts[:2]],
                'holdout_span_frames': [starts[2], starts[2]+block], 'context_excluded_frames': taps,
                'no_effect': no_effect},
            'limits': 'Approximate stationary linear effect. Does not identify the original plugin, nonlinear processing, modulation, or vocal content lost by the model.'}


def save_capture(path, params):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name+'.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(params, stream, allow_nan=False)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def capture_reverb(dry_path, wet_path, param_output_path, **settings):
    dry, sr = sf.read(dry_path, always_2d=True)
    wet, wet_sr = sf.read(wet_path, always_2d=True)
    if sr != wet_sr:
        raise ValueError('Dry and effect sample rates differ')
    try:
        params = fit_effect_ir(dry, wet, sr, **settings)
    except ValueError as error:
        # Overwrite an old capture with explicit failure so it cannot be reused.
        params = {'schema': 2, 'revision': REVISION, 'sample_rate': sr,
                  'valid_for_restore': False, 'rejection_reasons': [str(error)]}
    save_capture(param_output_path, params)
    return str(param_output_path)


def restore_reverb(dry_path, param_path, output_path, wet_level=1., keep_tail=False, allow_unvalidated=False):
    params = json.loads(Path(param_path).read_text(encoding='utf-8'))
    if params.get('schema') != 2:
        raise InvalidImpulseResponse('Legacy IR requires fresh capture with the stereo pipeline')
    diagnostic = params.get('diagnostics', {})
    requested = params.get('restore_requested') and diagnostic.get('dry_rms', 0) >= 1e-7 and diagnostic.get('maximum_frequency_gain', float('inf')) <= 8
    if not params.get('valid_for_restore') and not allow_unvalidated and not requested:
        raise InvalidImpulseResponse('; '.join(params.get('rejection_reasons', ['Unvalidated IR'])))
    if 'impulse_response' not in params:
        raise InvalidImpulseResponse('No fitted response available')
    if not 0 <= wet_level <= 2:
        raise ValueError('wet_level must be between zero and two')
    dry, sr = sf.read(dry_path, always_2d=True)
    dry = audio_array(dry)
    impulse = audio_array(params['impulse_response'])
    ir_sr = int(params['sample_rate'])
    if ir_sr != sr:
        divisor = math.gcd(ir_sr, sr)
        # A discrete convolution kernel scales inversely with the new sample rate.
        impulse = resample_poly(impulse, sr//divisor, ir_sr//divisor, axis=0) * ir_sr/sr
    effect = convolve_effect(dry, impulse, keep_tail)
    if dry.shape[1] == 1 and effect.shape[1] == 2:
        dry = np.repeat(dry, 2, axis=1)
    dry = np.pad(dry, ((0, len(effect)-len(dry)), (0, 0)))
    output = dry + wet_level * effect
    if not np.isfinite(output).all():
        raise InvalidImpulseResponse('Non-finite reconstruction')
    sf.write(output_path, output, sr, subtype='FLOAT')
    return str(output_path)
