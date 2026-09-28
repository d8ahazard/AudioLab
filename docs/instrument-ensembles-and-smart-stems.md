# Instrument ensembles and Smart Stems

## Listener direction, September 24, 2026

Freak: the listener found Demucs 6 and BS-RoFormer SW on the new instrumental similarly good. Compare combinations before choosing a production recipe. Mega 53 was not useful on this mix; retain it as an opt-in experiment, outside the default separation pipeline. This evaluation does not promote an ensemble or turn Mega on automatically.

## Listening pages

- `outputs/separation_v4_validation/freak-ensembles/index.html`: four instrument estimates (two original models, equal mean, consensus cleanup); five drum estimates (two original routes, equal mean, cleaned combination, and a fresh DrumSep pass on the cleaned blended drum parent). The sixth model output is the vocal/residual bucket, not a replacement for the approved hybrid lead vocal. Five instrumental children are summed for the reconstruction comparisons; the unassigned remainder is supplied separately, never automatically filled back in.
- `outputs/separation_v4_validation/orchestral-mega/index.html`: six independent context-padded excerpts from the Musopen public-domain recording of Rimsky-Korsakov's Russian Easter Festival Overture. All 53 model outputs retained. Recording source, download checksum and public-domain metadata are in `recording.json`. The score covers strings, winds, brass, harp and varied percussion; it is not an example of all 53 classes being present.
- `outputs/separation_v4_validation/smart-stems-review/index.html`: activity audit of Freak's earlier 53 outputs. The review controls collect winner/notes, including “No useful stem,” to distinguish faint useful material from musical leakage.

Run `scripts/compare_instrument_ensembles.py freak`, `orchestra`, or `smart` with the working Python environment. Predictions reuse the existing checksum-pinned AudioSeparator adapters and use Balanced inference. Raw outputs and original source projects are preserved. Page exports use one common downward gain per page; activity is measured before that gain.

## Blend candidate

`consensus_blend()` operates on aligned stereo estimates. It averages complex STFT estimates except where destructive cancellation would reduce their mean magnitude by more than 6 dB; there it retains the stronger estimate, with the same selection for both channels. The optional cleanup attenuates only amplitude disagreement where the other instrument estimates dominate, smooths that evidence, and caps attenuation at 3 dB. This can still lose content or retain bleed, hence the plain average, original estimates and reconstruction auditions. It is not a learned instrument classifier and has not been promoted as the winner.

## Smart Stems decisions

1. **Hide safely:** digital silence or extremely faint stationary broadband noise, using the existing conservative detector. Tonal frames or isolated onsets veto noise-only hiding. Hidden audio remains recoverable.
2. **Review, keep saved:** relative RMS below −30 dB and every 40 ms window below −20 dB relative to parent RMS. These are quiet candidates, not declarations of absent instruments. Production exports and downstream behavior retain them. The experiment pages collapse sections containing only review estimates; their checkbox or instrument search reveals them.
3. **Keep:** remaining estimates. This means measurable activity, not verified instrument identity. A fake guitar prediction containing quiet strings can pass an activity test. We need listening labels or a separately validated instrument-presence classifier to exclude musical cross-talk safely.

New separation manifests include maximum window-relative energy, active-window fraction, review status and reason. AudioLab's Smart Stems recovery panel now has **Review quiet stems** and audio preview. Existing hidden-stem restoration remains available. The pipeline revision is advanced so old manifests without measurements are not mistaken for new results. Restart the application to load the changed UI.

The first Freak audits classify 59 of 63 ensemble estimates as keep and 4 as review; the old Mega outputs classify 29 as keep and 24 as review. None meet the strict automatic silence/noise rule. That is intentional evidence against replacing instrument detection with an aggressive volume gate.

Validation: 63 unit tests passed, including agreement preservation, stereo preservation, cancellation protection, controlled bleed reduction, retention of sparse/quiet/noisy content, routing, cache/recovery and existing reverb behavior. These tests do not establish perceptual superiority on real music.

Sources: [Musopen recording collection and public-domain metadata](https://archive.org/details/MusopenCollectionAsFlac), [Boston Symphony orchestration notes](https://www.bso.org/works/rimsky-korsakov-russian-easter-festival-overture), [Mega author release](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21).
