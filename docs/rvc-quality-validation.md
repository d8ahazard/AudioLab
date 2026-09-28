# RVC quality validation: Huxlxy, Shinedown666 and Chester Bennington

## Current scope: existing RVC versus trained Seed-VC

The user paused V3 and replaced the broad listening sweep with two A/B passages per singer: Huxlxy / Preacher Man, Shinedown666 / Cracks in my Calm, and Chester Bennington / The Emptiness Machine. Chester uses the newest local export, ChesterRedux (300 epochs, 48 kHz V2), its matching index, and six original training vocal files. The Emptiness Machine is excluded from training and reference audio and receives the same maximum hybrid-cleaned/karaoke separation as the other evaluation songs.

`scripts/rvc_seed_ab.py` prepares Chester's isolated data, renders the matched comparisons, and publishes `listening/seed-ab.html`. Each pair contains the existing RVC model and a singer-specific 500-step Seed-VC singing fine-tune. No zero-shot candidate, V3, YingMusic or Vevo2 appears in this round. These are bounded fine-tuning candidates, not proven optimal checkpoints. Shinedown's Seed model uses refreshed training vocals, so the comparison evaluates the resulting systems rather than isolating architecture from dataset changes. Existing RVC checkpoints and all original recordings remain untouched.

`focused_seed_ab.json` blocks the earlier broad queue, and `v3/PAUSED` blocks V3 inference. Previously completed V3 work and other experimental artifacts remain available. The old quick-round preferences are saved separately; new A/B choices use separate browser storage and `seed-ab-reveal.json`.

This study separates three questions: better source stems, better conversion, and better training data. All outputs live in `D:/AI-outputs/AudioLab/rvc_quality_validation`. Historical projects, training stems, models, and indexes are not replaced. No candidate becomes a production default automatically.

## Fixed corpus

| Singer | Frozen baseline | Test mix |
|---|---|---|
| Huxlxy | `D:/models/AudioLab/trained/huxlxy_v2_v500.pth`, matching index | Preacher Man Sloppy AI Remix, project `71ff2927` |
| Shinedown666 | Inference export of `voices/Shinedown666/saves/G_e75.pth`, matching index | Cracks in my Calmv2 Cover 1, project `10f2b696` |
| Chester Bennington | `D:/models/AudioLab/trained/ChesterRedux.pth`, epoch 300, matching index | The Emptiness Machine, original downloaded mix |

All three baselines are pitch-guided V2 at 48 kHz. The Hux epoch-315 export is a secondary control from the earlier study. The older ShineDown/ ShineDown3 runs are not substituted for Shinedown666. `corpus.json` records paths, SHA-256 hashes, source mixes, old stems, and all nine protected Hux original training files. `focused_seed_ab.json` adds Chester model provenance. `models/Shinedown666_e75.json` records inference-export provenance; actual V2 strict loading is checked before export succeeds.

The seven refreshed Shinedown songs are DEVIL, Cut the Cord, MONSTERS, Second Chance, Simple Man, Sound of Madness, and State of My Head. Both evaluation songs are excluded from adaptation and target reference selection. Simple Man is held out from the new adaptation, but was present in the historical epoch-75 training data: its scores are regression checks, not unseen-song generalization evidence. Hux's acid_rain_vox is reserved from Seed adaptation; it likewise may have appeared in historical RVC training.

## Separation and controls

The exact recipe is `hybrid_cleaned`, `maximum`, Frazer–Becruily karaoke backing separation, one backing layer, smart stems off, and no extra reverb/echo/noise/crowd removal. Each separation retains full vocals, lead vocals, backing vocals, and instrumental. Per-song completion manifests record source and output hashes, settings, and wall time.

Four fixed 12-second passages per evaluation song are selected by energy, one per quarter. `excerpts.json` freezes their offsets. Energy screening does **not** establish coverage of consonants, sustained notes, register changes, or expressive delivery; listen to the passages and annotate that coverage before treating the test set as complete.

The frozen-model old/new-stem control uses the same offsets, checkpoint, retrieval rate, protection, seed, and pitch method. The Hux epoch comparison uses the refreshed stem. Independent Shinedown original/refreshed adaptations start from identical epoch-75 G/D weights with fresh optimizers, LR 0.00002, batch 4, seed 20260924, and 25 adaptation epochs. Periodic checkpoints and held-out audio are retained. Their conversion control uses retrieval **off**, so training-data effects are not obscured by a changed retrieval database. The old index is not represented as a new refreshed-data index.

Additional Sucial Fused dereverb is auditioned on separate 20-second training excerpts. It does not replace the untreated dataset. Listen for lost rasp, breath, consonants, and decays before deciding whether to make a further dry-data training arm.

## Backends and research interpretation

* V2: sweep index rate 0, .25, .5, .75, 1; compare protection 0, .2, .33, .5 at index .5; compare RMVPE+, RMVPE, CREPE, FCPE. FCPE is optional (`requirements-rvc-evaluation.txt`). These settings can trade resemblance against articulation; none is assumed to win.
* [Seed-VC](https://github.com/Plachtaa/seed-vc): use the pitch-conditioned 44.1 kHz singing checkpoint, 50 diffusion steps, length adjustment 1, pitch shift 0, automatic pitch adjustment off. Its published results motivate a trial, not a universal superiority claim. Local fine-tuning uses the official Trainer, 500 pilot steps, batch 2, and training-only 10-second crops. The external checkout has one explicit retention patch: `keep_all_checkpoints` prevents its normal checkpoint deletion.
* [YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC): evaluate the official full checkpoint using the same source/reference clips, 50 steps, and zero pitch shift. Its singing-specific training is a candidate improvement over a general conversion model; actual results remain case-specific.
* [Vevo2](https://github.com/open-mmlab/Amphion/tree/main/models/svc/vevo2): compare source-delivery preservation with target-style conversion. The style experiment uses source prosody, explicit target duration, and no automatic pitch shift. Its [checkpoint license](https://huggingface.co/RMSnow/Vevo2) is CC-BY-NC-ND 4.0; this study uses inference, not Vevo fine-tuning or redistribution. Source and reference lyrics are automatically transcribed locally and marked **unreviewed**; transcript errors confound this experiment and must be corrected before a firm conclusion.

The original V3 stack is feasible as research, but expanding a tensor is not sufficient to transfer a trained voice. A Whisper/HuBERT fusion or a different vocoder needs compatible learned conditioning and/or retraining. The repaired `v2_compatible` backbone preserves real V2 encoder, posterior, flow, NSF decoder and discriminator topology, and explicitly rejects unsupported dual-encoder conditioning. Legacy metadata retains its legacy topology choice. New configurations validate STFT bins, upsample product versus hop, and whole-frame segments. Content/pitch streams use the configured audio timebase. The native text dataset uses file-associated timestamped text for its crop; the compatibility loader rejects ambiguous project-wide lyrics. Missing-text rows bypass cross attention safely.

Both real singers passed the transfer audit: 560 core tensors copied, all equal to the training checkpoint; same-input waveform error below 0.0000006 after matching the V2 export precision and RNG. This proves the transfer contract, **not** text quality or complete-pipeline equivalence. V3 pipeline controls still include its own retrieval scaling, silence gate, loudness handling, and chunking; `effective_settings` records these for new renders. Text-off controls use transferred voice weights.

A separate Hux lyric-adapter feasibility pilot uses two training-only crops per original training song (excluding acid_rain), locally transcribed word timestamps, and 200 updates of only the text encoder/cross-attention layers. Its objective is the V2 posterior-to-prior KL loss plus a residual penalty; the vocoder and all voice-backbone weights stay frozen and are checked afterward. The same adapter checkpoint is auditioned with text off, corresponding lyrics, and unrelated reference lyrics. The new `char_v1` vocabulary is explicit in checkpoint metadata and shared by training/inference; legacy checkpoints retain their previous token IDs. This is a small, unreviewed-ASR pilot, not proof that lyrics repair singing or capture delivery.

## Application interface

Clone now offers **Singing conversion**, alongside existing RVC/OpenVoice/TTS paths. Controls explicitly select backend, target reference, optional checkpoint/config, conversion mode, source lyrics, reference words, and seed. Within that experimental mode, Vevo2 target delivery is the default; Seed-VC/YingMusic require preserve mode. Unsupported capabilities raise a clear error instead of silently ignoring settings. Standard RVC stays the default cloning method and keeps V2 model compatibility.

External engines run in `D:/venvs/audiolab-svc`; AudioLab's working CUDA Torch installation is shared read-only through a `.pth` entry. Source checkouts are under `E:/dev/AudioLab-experiments`; checkpoint/cache junctions point to D:. Configure `AUDIOLAB_SVC_PYTHON` and `AUDIOLAB_SVC_REPOS` to relocate the engines. External requirements use Transformers 4.46.3 and their needed dependencies without installing upstream's old Torch pins into AudioLab. See the install logs in the study output for the actual environment changes. Optional PyAV submodules are imported explicitly before torchvision's video probe.

`scripts/rvc_backend_worker.py` accepts a JSON request and writes `result.json` only after producing finite, nonempty audio. It records source/model/reference/config and output hashes, repository revision, seed/settings, runtime, peak VRAM and basic audio statistics. A file lock serializes conversion jobs to reduce memory contention. Training/separation can still share the GPU, so current wall times include loading and contention and are not speed benchmarks.

## Commands and artifacts

Run from `E:/dev/AudioLab` with `venv/Scripts/python.exe` unless noted:

```powershell
python scripts/benchmark_rvc_quality.py prepare
python scripts/benchmark_rvc_quality.py export
python scripts/benchmark_rvc_quality.py separate
python scripts/benchmark_rvc_quality.py excerpts
python scripts/benchmark_rvc_quality.py references
python scripts/benchmark_rvc_quality.py v2
python scripts/benchmark_rvc_quality.py seed_vc
python scripts/benchmark_rvc_quality.py yingmusic
python scripts/transcribe_rvc_quality.py
python scripts/benchmark_rvc_quality.py vevo2
python scripts/benchmark_rvc_quality.py vevo2_style
python scripts/rvc_stem_controls.py
python scripts/validate_v3_transfer.py
python scripts/rvc_v3_controls.py
python scripts/train_v3_text_quality.py
python scripts/rvc_text_ablation.py
python scripts/train_rvc_quality.py original
python scripts/train_rvc_quality.py refreshed
python scripts/train_rvc_quality.py seed-data
D:/venvs/audiolab-svc/Scripts/python.exe scripts/train_seed_quality.py huxlxy
D:/venvs/audiolab-svc/Scripts/python.exe scripts/train_seed_quality.py shinedown
python scripts/rvc_adapted_controls.py
python scripts/rvc_dereverb_controls.py
python scripts/rvc_full_song_controls.py
python scripts/report_rvc_quality.py --metrics
```

The finite `continue_rvc_evaluation.py comparisons` and `refreshed` lanes record completion/failure in `stages/`; the refreshed lane waits for the seven training separations. Do not start duplicate lanes while an existing lane is active. To recompute a stage after an input change, run the underlying command; baseline/external jobs check input/output hashes when resuming. Training completion markers indicate execution, not a quality endorsement.

The user requested **two tests per song, three candidates per test, one winner**. Run `python scripts/rvc_quick_listening.py` to publish this fixed four-test round at `listening/index.html` and `quick.html`. It uses passages 0 and 2 for each song. Hux passage 0 retains original labels B, C and F from the user's retained set; G is omitted to meet the three-candidate limit, not declared inferior. The same three settings are compared on both passages and voices. Prior rejections A, D and E are saved in `feedback.json`. Picks persist locally and can be copied into chat. `quick-reveal-key.json` maps this frozen round; subsequent background renders cannot add samples to it.

The full diagnostic report moves to `listening/diagnostics.html` once the quick round exists. Vocal RMS is matched; each passage uses one common playback gain and the same instrumental. Padding/trimming is only for the listening copy; original backend renders remain intact. `reveal-key.json` is deliberately separate. `metrics.json` records duration error, unshifted RMVPE pitch disagreement and envelope lag as supporting diagnostics. They are not substitutes for naturalness or identity judgments. Avoid reading the keys until choices are saved.

Complete-song controls are candidates for confirmation, not automatically selected winners. Promote only after blinded passage scores and complete-song checks establish a useful gain without damaging melody, articulation, or placement. Generalization beyond these two voices and songs remains untested.

Validation commands:

```powershell
python -m unittest testing.unit.modules.test_rvc_v3_contracts testing.unit.modules.test_svc_backends
python scripts/validate_v3_transfer.py
```
