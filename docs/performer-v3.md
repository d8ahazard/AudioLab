# Performer V3: implementation and research contract

The goal is performer-specific vocal delivery within the original song structure: identity, pronunciation, articulation, stress, timing, breath and singing expression. Artist identity is a conditioning key; race is neither a label nor a control. The existing RVC V2 pipeline remains the sound-quality baseline.

## Status and scope

This implementation starts **M0 and M1**, not an already-trained V3 release. The new `modules/rvc_v3/performer` package is independent of the paused legacy text-attention pilot. Existing checkpoints and indices are read and hashed, never overwritten. The legacy pause marker remains in place.

Implemented components:

- Versioned request/profile manifests, input hashes, source word timing, phrase windows and explicit edit windows.
- A constrained integer duration allocator that rejects infeasible budgets.
- Direct delegation to the existing V2 conversion worker for `preserve` or strength zero.
- An isolated CosyVoice3 pronunciation-guide process, then generated-guide HuBERT features rendered through frozen V2 with the source musical pitch structure.
- Six-layer, width-256 planner and acoustic bridge definitions. These are **untrained hypotheses**. The worker rejects `stage=learned`; random weights cannot masquerade as performer learning.
- Dataset review gates, candidate inventories and coverage reports that distinguish available minutes from reviewed target-only minutes.
- An immutable two-passage-per-target benchmark runner and success/failure ledger.
- Experimental Performer V3 clone selection. Until M1 passes, the guide branch supports strengths zero or one only.

Not yet established: learned pronunciation/flow, a trained acoustic bridge, lyric-edit quality, sustained singing quality, full-song consistency or generalization. Aligned interactive lyric editing and continuous delivery strength belong to later milestones. Do not describe the current guide control as learned artist delivery.

## Evidence and hypotheses

| Source | Established contribution | Local use / limitation |
|---|---|---|
| [Vevo](https://arxiv.org/abs/2502.07243) | Content/style generation separated from timbre-conditioned acoustics | Architectural motivation, not evidence of beating our models |
| [Vevo2](https://huggingface.co/RMSnow/Vevo2) | Singing style and lyric-edit inference | Comparison only; published checkpoint CC-BY-NC-ND |
| [StyleStream](https://github.com/Berkeley-Speech-Group/StyleStream) | Separating source style from content | Research evidence; public release lacks the arbitrary-target style encoder |
| [CosyVoice3 base](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | Released text-to-acoustic foundation | Pronunciation guide, not a demonstrated singing renderer |
| [CosyVoice3 training recipe](https://github.com/QwenAudio/CosyVoice/blob/main/examples/libritts/cosyvoice3/run.sh) | LLM training path | Adapt token-model attention with LoRA first; freeze acoustic decoder |

Our previous listening study favored RVC overall. That motivates retaining its renderer, not assuming newer generators are inherently inferior. The original lyric adapter reconstructs source RVC features and does not supervise the desired transformation.

The key hypothesis is that a performer-conditioned planner can generate changed articulation and timing, and a trained bridge can translate that into the 768-dimensional RVC content space without losing identity or fidelity. A second hypothesis is that speech-derived guides can support sustained singing when aligned and conditioned appropriately. Both require falsification tests.

## Architecture and feature contracts

Source vocals are analyzed into words, phrase windows, voicing, coarse musical contour, energy and structural anchors. Expressive generation must not receive unrestricted source content features: that makes copying the original delivery the easiest solution.

Performer profiles contain the immutable RVC checkpoint/index, curated target recordings, reviewed aligned text, behavior coverage, training-only reference memory and later learned adapters. Memory should retrieve phrase examples by phonetic context, delivery mode, register, intensity and density. It must not compress every behavior into one averaged embedding.

The planned delivery model takes desired phonetic content, structure, performer conditioning and seed. Separate rap and singing heads predict duration logits, stress, breath, pitch gestures, energy, articulation and texture. The bridge takes aligned guide features, planned expression and retrieved target features and predicts 768-dimensional RVC content. The core RVC encoder, flow and decoder remain frozen initially.

Time is represented in seconds at public boundaries. The initial renderers explicitly verify their checkpoint's sample rate / upsample product equals the expected 100-Hz frame rate. The native hop is read from the checkpoint, not assumed universally. Output duration is the original sample timeline. Frame-grid rounding is bounded to one frame, with no large truncation used to hide a failed fit.

**M1 approximation:** the untrained control allocates time by voiced/unvoiced guide runs, not phones. It limits compression and extension and places source pitch under generated content. This deliberately cheap test cannot establish target cadence. If stretched speech, lost consonants or excessive holds remain audible, diagnose guide/timing/renderer separately before training. Automatic transcript disagreement is recorded for review, never declared successful word preservation.

## Data and contamination controls

Initial targets: Tupac1K, huxlxy_v2_v500, Shinedown666 epoch-75 export, ChesterRedux epoch 300. Development sources: Lose Yourself, Preacher Man remix, Cracks in my Calm, The Emptiness Machine. These are demonstrations, **not unseen-song generalization tests**. Freeze a second source song per target before claims of generalization.

Original vocal files are immutable. Candidate records have recording-level train/validation/test splits. Selected segments require separate target-only, transcript and alignment review. Excluded guests/hooks/doubles have recorded reasons. No reviewed segments means no adaptation. Canonical text and observed pronunciation must remain distinct when phonetic annotation is added.

Huxlxy's original isolated stems have user-confirmed provenance. Other folders can contain featured performers and backing vocals; automatic diarization alone is not proof of target identity. Reference crops are candidate-only until reviewed. Exact hashes catch identical files; acoustic duplicate/alternate-version detection remains necessary before training and cannot be claimed from byte hashes alone.

Coverage must report phonetic contexts, register, intensity, density, vowels, consonants, texture and transitions. The 30–60 minute target is a collection goal, not demonstrated sufficiency. Repeat epochs do not repair missing coverage. Use untreated separations as the primary preserved artifact and compare additional cleanup separately.

## Training protocol after feasibility

1. Generate guides for reviewed real target phrases. Freeze whole-recording splits and reference eligibility first.
2. Adapt only the CosyVoice token-model attention: rank 16, alpha 32, dropout .05, LR 2e-5; batch 1, accumulation 16, gradient checkpointing. Verify exact injected modules and trainable parameter manifest before running.
3. Supervise the planner from real target durations, stress, breaths, pitch gestures and energy. Preserve phrase/note anchors, not the source's exact fine timing.
4. Train the acoustic bridge against real target HuBERT features for the same words, then waveform reconstruction through frozen RVC. Use bounded crops for renderer gradients.
5. Jointly refine planner/bridge with text consistency, budget accuracy, feature reconstruction, pitch/energy and spectral losses. Feature matching requires an explicit frozen feature extractor or compatible discriminator, not a fictitious loss against an absent model.
6. Only if correct planned delivery is consistently suppressed by the frozen renderer, test small residual prior/flow adapters. A new vocoder requires a revised plan based on that evidence.

Training is not launched by importing model definitions or selecting the experimental UI. Maximum initial runs: 6,000 updates, validate every 500, stop after three plateaus. Each run needs resumable optimizer/RNG state, dataset/model/code hashes and measured calibration costs. No silent failed-run restart or model pruning.

## Local execution

Models: `D:/models/AudioLab-performer/CosyVoice3`. Isolated guide environment: `D:/venvs/audiolab-performer`. Upstream source: `E:/dev/AudioLab-experiments/CosyVoice`. Study artifacts: `D:/AI-outputs/AudioLab/performer_v3_validation`.

The guide environment reuses the existing CUDA-capable Torch runtime through a `.pth` entry and pins Transformers 4.51.3 / PEFT 0.15.2. Record the resolved environment before training; upstream requirements must not downgrade the production CUDA runtime. All heavy study workers use the existing study conversion lock. Foundation models run in separate processes and exit before RVC rendering. Cached features stay on D: and datasets must use bounded reads.

From the repository in PowerShell:

```powershell
& ./venv/Scripts/python.exe scripts/prepare_performer_study.py
& ./venv/Scripts/python.exe -m pytest testing/unit/modules/test_performer_v3.py -q
& ./venv/Scripts/python.exe scripts/benchmark_performer_v3.py --round m1-001 --voices tupac huxlxy --passages 0
```

Separate the Tupac source and freeze excerpts first using `benchmark_rvc_quality.py` with `--root D:/AI-outputs/AudioLab/performer_v3_validation`. A failed attempt is retained; use a new round name after fixing its cause. Never replace previously scored audio under an existing label.

The Python service accepts source_audio, performer_profile, output_dir, delivery_mode, source_transcript, target_lyrics or phrase_edits, delivery_strength, seed and stage. Phrase edits are `{start, end, text}` in absolute seconds. Corrected preview text is forced-aligned with the installed WhisperX / Wav2Vec2 path, with missing or low-confidence words rejected. This path is bounded to 20 seconds; corrected full-song text still requires phrase alignment. The final CTC frame may overrun the waveform by a stride; that bounded rounding is recorded and clamped to the original samples. Replacement words are explicit and separate from source transcription.

## Initial implementation findings

- The first transcript-conditioned rap guide and RVC render matched desired words under automatic ASR, at exactly 12 seconds after assembly. Human naturalness, identity and delivery are still unscored.
- The first Huxlxy short-phrase guide inserted a word and muddled another phrase before RVC rendering. The upstream implementation warned that the desired text was short relative to the reference transcript.
- An explicit audio-only reference control (`guide_conditioning=audio_only`, using the upstream cross-lingual interface with CosyVoice3's required end-of-prompt marker) removed those observed errors in the first Huxlxy clip. This isolates a useful guide-conditioning choice; it does **not** prove learned artist pronunciation.
- The first rap guide process measured about 3.74 GB peak PyTorch GPU allocation, 7.60 GB peak process working set and 68.6 seconds including load/generation/hash work. These are inference measurements, not estimates of adapter training cost. ONNX feature extraction used its available CPU provider.
- Preserve-mode and direct V2 invocation share the same worker/settings, sample rate and shape. A contemporary rerun comparison measured RMS difference about 0.0000405 and maximum difference 0.000458. Routing is exact; these observations do not establish bit-identical GPU determinism. Historical baseline comparison also differs numerically and is recorded separately.
- Bootstrap failures (missing dependency, an omitted CosyVoice3 prompt marker in the new audio-only caller, and alignment boundary handling) are retained as implementation failures. They are not evidence against the research architecture.

The round runner supports `--modes preserve performer` and `--guide-conditioning transcript|audio_only`. Keep diagnostic choices in saved requests. Do not silently fall back between these conditions after a phrase fails. Strict listening publication requires matching automatic words or a hash-bound manual word review. The explicit `--review-disputed-words` option admits discrepancies for human review only, after comparison against the same recognizer on V2. It never marks lyrics correct or passes M1. Automatic recognition also misreads the baseline and the source lyrics, so zero ASR error is not a reliable musical-quality gate.

## Executed feasibility round — September 24, 2026

Eight comparisons have been rendered: two passages each for Tupac, Huxlxy, Shinedown and Chester. The final controlled round is `m1-proportional-001`, reusing the exact audio-only pronunciation guides and V2 baselines from `m1-audio-only-002`. A corrected bounded proportional timing allocator preserves native run durations when no rescaling is needed; the previous allocator redistributed them unnecessarily. This correction removed an automatically detected extra word in the second Huxlxy passage. It does not establish natural singing or performer-specific delivery.

The [A/B review](http://127.0.0.1:8769/performer-m1-proportional-001-pcm24/) contains eight pairs, with fixed letter mappings, original render hashes and separate RMS-matched PCM24 playback files. No listening scores have been entered. Some source transcripts remain uncertain, especially the original Huxlxy and Shinedown lyrics. User lyric text is needed to settle those words. Browser automation loaded the review but the in-app tab crashed during playback verification; actual browser playback is therefore not verified.

Twenty-two targeted performer, legacy V3 and SVC tests passed. Hash verification confirmed all 40 original/profile assets unchanged. Reports, failures, resource measurements, candidate data inventories and exact runtime snapshots are retained under `D:/AI-outputs/AudioLab/performer_v3_validation`.

M0 remains partial: benchmark and compatibility infrastructure exist, but reviewed/aligned training minutes are zero. M1 has executable rap and singing feasibility candidates, with listening and word-correctness gates pending. Planner and learned bridge definitions exist; no performer adapters or learned acoustic bridge have been trained. M2–M5 are not complete. Experimental UI integration requires a normal AudioLab restart; the existing application was not interrupted.

Reproduce a new immutable round (choose an unused identifier):

```powershell
& ./venv/Scripts/python.exe scripts/benchmark_performer_v3.py --round m1-new --voices tupac huxlxy shinedown chester --passages 0 2 --modes preserve performer --guide-conditioning audio_only
& ./venv/Scripts/python.exe scripts/listen_performer_v3.py --round m1-new --review-disputed-words
```

## Gates and evaluation

### Recorded user verdict on the first M1 round

The user selected existing RVC in **all eight comparisons**: Tupac B/B, Huxlxy B/A, Shinedown A/A, Chester A/A. The feedback credits repetition of the words but reports strange accents and lost target identity. Choices are bound to the unchanged mapping and candidate hashes in `rounds/m1-proportional-001/listening-review-001.json`. This supersedes the pending-listening status above: the tested configuration **failed M1**. Word repetition is not a formal exact-lyrics pass.

This result rejects the current untrained guide/timing/frozen-renderer combination as a useful output. It does not isolate the culprit or test the proposed learned bridge. Do not proceed to long adaptation runs or another parameter-sweep listening round on this evidence. Next diagnosis should compare real target audio through the same renderer path (without timing changes), generated guides before/after timing, and matched retrieval/protection conditions. These internal controls separate renderer-path regressions, guide pronunciation leakage and timing damage. Only after that diagnosis should a small real-target reconstruction experiment test whether a learned bridge can retain identity; no presumption that additional training will fix it.

| Gate | Required evidence |
|---|---|
| M0 | Reviewed target-only data, corrected alignment, frozen benchmarks, exact V2 routing |
| M1 | Rap **and** singing guides render intelligible desired words, fit windows, and do not become stretched speech |
| M2 | Learned planner/bridge improves target delivery over V2 and untrained guide control, without material identity/intelligibility loss |
| M3 | Unseen word/phrase replacements succeed; overfull edits are rejected |
| M4 | Full-song benefit, stable performer identity, no cumulative drift or seams |
| M5 | At least one rap and one singing profile pass; others remain experimental |

Internally compare V2, untrained guide+RVC, learned planner+bridge and removed/shuffled target conditioning. Weak change when conditioning is removed means the performer model has not learned its purpose. For lyric edits the untrained guide is the relevant baseline; V2 is not a text editor.

Human evaluation stays at two passages per performer, A/B only, overall preference and optional reason. Internally reject broken output before requesting listening. Freeze mappings and SHA256s. Record text errors (with manual disputes), phone timing, stress, pauses, structural melody anchors, identity diagnostic, clipping/seams, runtime and memory. Pitch gestures deliberately changed by the planner are not automatically pitch errors. Check nearest training examples for copying before release.

Failure is a research result. M1 singing failure cannot be averaged away by rap success. M2 failing to beat the simple guide means stop that branch rather than celebrate extra complexity.
