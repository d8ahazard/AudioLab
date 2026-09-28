# Freak instrument screening

Run `venv/Scripts/python.exe scripts/compare_freak_instruments.py` from the repository root.
The experiment writes only to `outputs/separation_v4_validation/freak-instruments` and the shared model cache. It does not change application defaults or the source project.

The local source is `outputs/process/Freak OG_0a636df0/source/Freak OG.wav`. Listening excerpts start at 12, 65, 130 and 210 seconds and last 12 seconds. Each inference input is processed independently with four seconds of source context before and after the audition interval. The listening montage joins these four excerpts; it is screening evidence, not a full-song validation.

## Candidates

| Candidate | Purpose |
| --- | --- |
| HTDemucs 6s | Existing six-stem baseline; both original mix and new instrumental input |
| BS-RoFormer SW | Six-stem challenger; both input routes. Existing checkpoint matches the published SW Fixed hash `24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e`; not a newly downloaded distinct model |
| Becruily MelBand Guitar | Dedicated guitar prediction from the new instrumental; author repository revision and hashes pinned in the runner, license unspecified by its model card |
| MDX23C DrumSep aufr33/jarredou | Six kit components from each six-stem model's isolated drum output: kick, snare, toms, hi-hat, ride, crash; use the actual installed YAML labels |
| MVSep Mega 53 v1 | April 20, 2026 broad-instrument challenger on the new instrumental; checkpoint/config verified against the author's GitHub release SHA256 digests |

The new instrumental is the equal waveform mean of Resurrection Vocals, Big Beta 7 and Becruily Instrumental accompaniment estimates, matching the instrumental branch of AudioLab's current cleaned hybrid. Vocal-target `other` outputs mean accompaniment in this stage only. No residual-fill heuristic is applied.

Balanced inference preserves author overlap and chunk settings, batch size 1 for MDXC, gain-preserving float audio, and autocast. Each prediction records package version, input hashes, model/config fingerprints, effective precision/overlap, runtime, peak allocated VRAM, and output hashes. A cached result is reused only when its identity and file hashes match.

The guitar model needs a comparison-process adapter: AudioSeparator 0.47.0's MelBand constructor accepts `mlp_expansion_factor` but does not pass it into its mask estimators. The runner scopes a constructor patch to this model to honor the author's expansion factor of 1. The checkpoint then loads with the correct tensor shapes. Installed package files and application behavior are unchanged.

Mega also needs a scoped adapter: the generic RoFormer validator caps `num_stems` at 16 even though the architecture supports 53 heads. For this checksum-pinned checkpoint only, the runner requires exactly 53 instead. Other validation remains enabled. `--mega-only` resumes this pass using the previously generated instrumental excerpts; `--report-only` regenerates the page without inference.

The page keeps quiet stems available and uses one downward-only gain shared across every player. Quiet outputs are not independently boosted. Per-instrument winner and notes are saved in browser local storage and can be exported as JSON.

## Interpretation

Completed September 24, 2026: 44 inference calls across the four context-padded excerpts, yielding 93 named listening files (48 seconds each), including all 53 Mega roles and both six-piece DrumSep routes. Verification checked every listening file's hash, stereo shape, sample rate, finite samples, playback headroom, and HTML audio link; the page JavaScript passed a syntax check. Shared playback gain is 0.8139378567. Mega's peak allocated CUDA memory was 8.37 GiB for these inputs; this is not a general full-song memory guarantee. Measurements and run provenance are in the page's `listening-manifest.json`; checks are in `verification.json`.

Prioritize separation from other instruments, intact attacks/decays, and missing musical events. An output label or nonzero RMS does not prove that instrument is present. There are no reference stems for this recording, so no reference SDR is reported.

Mega's family and child stems overlap (for example guitar/acoustic guitar/electric guitar or drums/kick/snare). Do not sum all 53 or combine aggregate drums with their kit children. Specialist outputs are alternative estimates, not automatically additive tracks. Keep an unassigned remainder in any later production mixing recipe, and evaluate a reconstructed mix before promotion.

The research also found SCNet XL IHF four-stem models and newer MVSep-hosted SCNet/MelBand drum specialists. They are not silently represented as tested here. Four-stem SCNet does not itself expand instrument taxonomy; hosted listings do not establish downloadable local weights. The present experiment covers public, locally runnable candidates without uploading the song.

## Primary sources

- [MVSep Mega 53 author release and caveats](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21)
- [Author's pretrained model catalog, including SCNet](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md)
- [Becruily guitar model](https://huggingface.co/becruily/mel-band-roformer-guitar)
- [MVSep drum separation models](https://mirror.mvsep.com/algorithms/29)
- [SW Fixed checkpoint provenance in the inference package](https://github.com/openmirlab/bs-roformer-infer)
