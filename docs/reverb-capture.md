# Stereo reverb capture

## Separation controls

`vocal_reverb` has three choices:

- **Keep wet** (default): leave the main vocal wet; no extra capture pass.
- **Dry vocals**: export Sucial Dereverb + Echo Fused's estimated dry main vocal.
- **Capture reverb**: run Sucial Fused for analysis, estimate and save the IR, and export the original wet main vocal. The dry estimate is temporary and is not substituted during cloning. Cloned vocals remain wet as explicitly requested by the user. Merge adds the saved response to the cloned vocal; this is additive to any effect already retained by cloning, not an automatic dry-and-restore cycle. Unchanged wet originals and already re-reverbed outputs are not processed again.

The user selected Sucial Fused after named listening. Its checkpoint/config are pinned and hashed in the AudioLab adapter; author CC-BY-NC-SA-4.0 metadata is retained. The new modes run it once on the main vocal and supersede legacy reverb/echo removal for that stem, avoiding duplicate cleanup. Legacy API options remain supported; unrelated cleanup and backing-vocal options are retained.

`handlers.reverb.extract_reverb(dry_path, wet_path, param_output_path)` now delegates to `modules.reverb_ir.capture_reverb`. **wet_path means the removed effect**, not the original dry-plus-wet recording. Use original minus estimated dry. Inputs must be aligned with the same sample rate and channel layout.

Schema 2 stores a causal, wet-only, channel-specific FIR plus fit settings, independent held-out diagnostics, and an explicit restore decision. It does not claim to identify a studio plugin or model time-varying/nonlinear effects. Default response length is 1.5 seconds; capture requires three disjoint blocks longer than twice that length. Two blocks fit the filter and the third validates it. No direct-vocal normalization, automatic delay shifting, or envelope-based RT60 guessing occurs.

The conservative automatic gate requires at least 10% reduction in held-out wet-effect squared error, no active channel worse than doing nothing, sufficient excitation and stable gain. Passing is not a perceptual quality guarantee. The Royals screen found only weak numerical support for one candidate; the other five fail. Following the user's listening approval, explicit **Capture reverb** records `restore_requested=true`: stored responses are applied despite a failed held-out score, provided an actual finite response, sufficient excitation and bounded gain exist. The failed score and warning are retained rather than relabeled as a validated fit. Legacy capture without this explicit request still uses the conservative gate.

`apply_reverb` adds the convolution to its input at `wet_level=1`. It preserves float gain, adapts sample rate, and supports mono clones with stereo IRs. Default `keep_tail=False` preserves stem duration; set it true for standalone rendering with the full convolution tail. Legacy schema and unusable captures raise `InvalidImpulseResponse`. The merge wrapper reports these and keeps its input stem. `allow_unvalidated=True` remains a diagnostic API override; the UI uses the recorded explicit capture request described above.

The separation pipeline captures once after the requested reverb and echo stages, before noise/crowd cleanup. Each source has its own `<source>__(Vocals).ir`; the manifest binds its hash to that source. Merge uses this binding and will not automatically use old shared `impulse_response.ir` files. Old files are preserved. Cache revision `hybrid-sucial-capture-4` invalidates old separation caches.

Run `testing/unit/modules/test_reverb_ir.py` for controlled fixtures, `testing/smoke_reverb_ir.py` for conservative-gate integration, and `testing/smoke_reverb_modes.py` for Sucial dry/capture output, exact wet preservation and a wet-clone stand-in's merge routing. No voice conversion is claimed by that smoke test. `scripts/compare_royals_reverb.py` builds named model comparisons, followed by `scripts/evaluate_royals_ir.py` for reconstruction previews. Results live under `outputs/separation_v4_validation/royals-reverb/`.
