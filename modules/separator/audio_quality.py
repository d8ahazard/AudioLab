"""Gain-preserving fusion and deliberately conservative stem activity analysis."""
from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter


def clean_hybrid_vocals(vocals, mix, instrumental):
    """Listener-selected residual-guided cleanup, capped at 3 dB per stereo bin."""
    vocals, mix, instrumental = [stereo(x) for x in (vocals, mix, instrumental)]
    if not (vocals.shape == mix.shape == instrumental.shape):
        raise ValueError("Hybrid inputs must have identical alignment and shape")
    if vocals.shape[-1] < 2048:
        return vocals.copy()  # Insufficient context: retain short vocal content.
    settings = dict(nperseg=2048, noverlap=1536)
    spectra = [signal.stft(x, **settings)[2] for x in (vocals, mix - instrumental, instrumental)]
    v, r, i = spectra
    vm, rm, im = [np.sqrt(np.mean(abs(x) ** 2, axis=0)) for x in spectra]
    disagreement = np.clip(1 - rm / (vm + 1e-10), 0, 1)
    dominance = np.clip((im / (vm + 1e-10) - 2) / 4, 0, 1)
    evidence = uniform_filter(disagreement * dominance, size=(3, 5), mode="nearest")
    mask = 1 - (1 - 10 ** (-3 / 20)) * evidence
    _, audio = signal.istft(v * mask[None], **settings)
    return stereo(audio, vocals.shape[-1])


def stereo(audio, length=None):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    if audio.ndim != 2 or audio.shape[0] not in (1, 2):
        raise ValueError(f"Expected channel-first mono/stereo audio, got {audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError("Non-finite audio samples")
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    if length is not None:
        audio = np.pad(audio[:, :length], ((0, 0), (0, max(0, length - audio.shape[-1]))))
    return audio


def consensus_blend(first, second, competitors=None, max_cut_db=3.0):
    """Experimental two-model blend; shared stereo decisions, bounded cleanup.

    Keep the stronger complex estimate where averaging would cancel by >6 dB.
    Else use the mean. Attenuate disagreement only where another stem dominates.
    This is an audition candidate, not a claim to identify instrumental bleed.
    """
    a, b = stereo(first), stereo(second)
    if a.shape != b.shape:
        raise ValueError("Ensemble inputs must have identical alignment and shape")
    if not 0 <= max_cut_db <= 3:
        raise ValueError("Cleanup must remain between 0 and 3 dB")
    if a.shape[-1] < 2048:
        return ((a + b) * .5).astype(np.float32)
    settings = dict(nperseg=2048, noverlap=1536)
    za, zb = [signal.stft(x, **settings)[2] for x in (a, b)]
    ma, mb = [np.sqrt(np.mean(abs(z)**2, axis=0)) for z in (za, zb)]
    z = (za + zb) * .5
    mean_mag = np.sqrt(np.mean(abs(z)**2, axis=0))
    cancelled = mean_mag < .5 * (ma + mb) * .5
    # Select both channels from the same estimate; never invert either channel.
    strong = np.where((ma >= mb)[None], za, zb)
    z = np.where(cancelled[None], strong, z)
    if competitors is not None:
        c = stereo(competitors)
        if c.shape != a.shape:
            raise ValueError("Competitor must have identical alignment and shape")
        zc = signal.stft(c, **settings)[2]
        cm = np.sqrt(np.mean(abs(zc)**2, axis=0))
        disagreement = abs(ma - mb) / (ma + mb + 1e-10)
        dominance = np.clip((cm / (np.maximum(ma, mb) + 1e-10) - 2) / 4, 0, 1)
        evidence = uniform_filter(disagreement * dominance, size=(3, 5), mode='nearest')
        z *= (1 - (1 - 10**(-max_cut_db/20)) * evidence)[None]
    _, result = signal.istft(z, **settings)
    return stereo(result, a.shape[-1])


def activity_review(audio, parent, sr):
    """Explain conservative hiding; separately flag quiet content for review.

    Review never changes the existing hide decision. Timbre/class membership
    cannot be inferred from energy: quiet harmonies and bleed both stay saved.
    """
    result = analyze_activity(audio, parent, sr).to_dict()
    x, p = stereo(audio), stereo(parent)
    frame = max(64, int(sr * .04))
    if x.shape[-1]:
        windows = np.pad(x, ((0, 0), (0, (-x.shape[-1]) % frame))).reshape(2, -1, frame)
        rms = np.sqrt(np.mean(windows.astype(np.float64)**2, axis=-1)).max(axis=0)
        parent_rms = np.sqrt(np.mean(p.astype(np.float64)**2)) if p.size else 0.
        relative = 20*np.log10(np.maximum(rms, 1e-12) / max(parent_rms, 1e-12))
        result['max_window_relative_db'] = float(relative.max())
        result['active_window_fraction'] = float(np.mean(relative > -35))
    else:
        result.update(max_window_relative_db=-240., active_window_fraction=0.)
    quiet = result['relative_db'] < -30 and result['max_window_relative_db'] < -20
    result['review'] = bool(quiet and not result['hidden'])
    result['display_status'] = 'hidden' if result['hidden'] else 'review' if quiet else 'keep'
    result['review_reason'] = 'quiet estimate; instrument presence unverified' if result['review'] else None
    return result


def fuse(tracks, algorithm="avg_wave", weights=None):
    """Select spectral bins jointly for both channels, never normalize stems up."""
    if not tracks:
        raise ValueError("Cannot fuse an empty ensemble")
    length = tracks[0].shape[-1]
    arrays = np.stack([stereo(t, length) for t in tracks])
    weights = np.asarray(weights if weights is not None else np.ones(len(tracks)), dtype=float)
    if len(weights) != len(tracks) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid ensemble weights")
    if algorithm == "avg_wave":
        return np.average(arrays, axis=0, weights=weights).astype(np.float32)
    if algorithm not in {"avg_complex", "median_magnitude", "min_magnitude", "max_magnitude"}:
        raise ValueError(f"Unknown fusion algorithm: {algorithm}")
    if length < 4:
        return np.average(arrays, axis=0, weights=weights).astype(np.float32)
    nfft = min(2048, length)
    _, _, spectra = signal.stft(arrays, nperseg=nfft, noverlap=3 * nfft // 4, axis=-1)
    if algorithm == "avg_complex":
        combined = np.average(spectra, axis=0, weights=weights)
    else:
        magnitude = np.sqrt(np.mean(np.abs(spectra) ** 2, axis=1))
        order = np.argsort(magnitude, axis=0)
        rank = {"min_magnitude": 0, "max_magnitude": -1, "median_magnitude": len(tracks) // 2}[algorithm]
        selected = order[rank]
        combined = np.take_along_axis(spectra, selected[None, None, ...], axis=0)[0]
    _, audio = signal.istft(combined, nperseg=nfft, noverlap=3 * nfft // 4)
    return stereo(audio, length)


@dataclass
class Activity:
    hidden: bool
    reason: str
    peak: float
    rms_db: float
    relative_db: float
    flatness: float = 0.0
    dynamic_db: float = 0.0

    def to_dict(self):
        return asdict(self)


def analyze_activity(audio, parent, sr, noise_filter=True):
    """Hide only digital silence or stationary, extremely faint broadband noise.

    A single energetic window, tonal frame or transient vetoes noise hiding.
    Measurements use channels independently (anti-phase audio must not cancel).
    """
    x = stereo(audio)
    p = stereo(parent)
    if not x.size:
        return Activity(True, "empty", 0.0, -240.0, -240.0)
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    parent_rms = float(np.sqrt(np.mean(p.astype(np.float64) ** 2))) if p.size else 0.0
    db = 20 * np.log10(max(rms, 1e-12))
    relative = 20 * np.log10(max(rms, 1e-12) / max(parent_rms, 1e-12))
    result = Activity(peak <= 1e-8, "digital silence" if peak <= 1e-8 else "content or uncertain", peak, float(db), float(relative))
    if result.hidden or not noise_filter or x.shape[-1] < sr or db > -65 or relative > -40:
        return result
    frame = max(64, int(sr * 0.04))
    frames = np.pad(x, ((0, 0), (0, (-x.shape[-1]) % frame))).reshape(2, -1, frame)
    energies = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=-1)).max(axis=0)
    dynamic = float(20 * np.log10(max(energies.max(), 1e-12) / max(np.median(energies), 1e-12)))
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(frame), axis=-1)) ** 2 + 1e-24
    flatness = np.exp(np.mean(np.log(spectrum), axis=-1)) / np.mean(spectrum, axis=-1)
    # A tonal frame in either channel or any isolated onset keeps the whole stem.
    result.flatness = float(np.percentile(flatness, 5))
    result.dynamic_db = dynamic
    if result.flatness > 0.35 and dynamic < 3 and peak < 0.003:
        result.hidden = True
        result.reason = "very faint stationary broadband noise"
    return result
