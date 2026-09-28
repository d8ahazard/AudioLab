# Selected stems and optional Mega extras

This supersedes the Smart Stems experiment and its September 24 review UI.

- Automatic silence/noise/quiet filtering is retired. Old `smart_stems` flags are ignored. Selected silent stems are still exported. The prior hidden-file recovery control remains, so previous outputs are not stranded.
- Individual instruments default to **consensus**: Demucs 6 and BS-RoFormer SW on the new instrumental, followed by the listener-selected consensus blend with gentle cleanup.
- With drum separation enabled, DrumSep runs independently on the original Demucs and SW drum estimates, then its corresponding outputs receive the selected blended-output cleanup. It does not substitute DrumSep on the already blended parent.
- `instrument_stems` chooses the exported instrument outputs. `drum_stems` chooses kit pieces. Models may internally predict more heads than are selected, because their inference architectures return all heads together. Unselected outputs are not exported. Existing vocal/instrumental and backing-vocal controls remain.
- `mega_stems` defaults to an empty list, disabling Mega entirely. The UI exposes 29 additional leaf categories. Vocals, bass/double bass, guitar variants, piano variants, kit outputs and aggregate strings/brass/winds/keys/percussion/bells are excluded. Specific percussion not supplied by DrumSep remains available, such as timpani, triangle, tambourine and congas.

Mega runs in a separate process to isolate its required 53-head validator adapter. Downloads remain checksum-pinned, source/model hashes and effective runs enter cache/manifest identity, and failures propagate. It uses the selected processing quality and CPU setting.

## Duplicate handling

Removing competing heads and aggregate/child categories prevents known taxonomy duplication. For selected Mega estimates, a very strict same-gain waveform comparison flags near-identical copies (relative squared error below 1e-8, confirmed over the full signal). Silence is not treated as duplicate evidence. There is no gain fitting or temporal alignment that might misclassify distinct unison instruments.

Detected copies are retained with a `Duplicate_of` filename suffix and a `duplicate_of` manifest field, and excluded from automatic mixing. Other Mega extras are also excluded from automatic mixing when the aggregate instrumental or `other` track is present. Users can audition the retained files and choose alternatives. The detector cannot guarantee the semantic identity of an instrument or eliminate every partially shared passage; it deliberately does not subtract shared musical content blindly.

The production instrument and drum blend calculations were compared to the 11 listener-selected Freak output roles using the cached individual predictions. Tests cover routing through both original drum parents, retaining silent selected outputs even with old Smart Stems flags, selection validation, forbidden Mega categories, duplicate handling and mixing exclusions. A real optional Mega run exports only the chosen flute and violin estimates. UI checkbox controls and API defaults are validated separately. Restart AudioLab to load the changed settings.
