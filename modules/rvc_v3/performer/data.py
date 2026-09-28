"""Review gates for adaptation; inventories are not silently treated as training data."""
from pathlib import Path
from .contracts import identity,validate_edits


def reviewed_training_segments(profile):
    """Fail closed: each selected segment needs speaker, text and timing review."""
    selected=[];seen={}
    for recording in profile['recordings']:
        source=recording['source']
        if identity(source['path'])!=source:raise ValueError('Recording changed since inventory')
        digest=source['sha256']
        if digest in seen and seen[digest]!=recording['split']:
            raise ValueError('Duplicate recording crosses dataset splits')
        seen[digest]=recording['split']
        for segment in recording['segments']:
            if segment.get('excluded'):continue
            validate_edits([dict(start=segment['start'],end=segment['end'],text=segment['text'])],recording['duration'])
            if not all(segment.get(k,False) for k in ('target_only_reviewed','transcript_reviewed','alignment_reviewed')):
                raise ValueError(f"Unreviewed segment in {recording['recording_id']}")
            if recording['split']=='train':selected.append(dict(recording=recording,segment=segment))
    if not selected:raise ValueError('No reviewed training segments; adaptation is gated')
    return selected


def coverage(profile):
    approved=[];uncertain=[]
    for record in profile['recordings']:
        for segment in record['segments']:
            if segment.get('excluded'):continue
            (approved if all(segment.get(k) for k in ('target_only_reviewed','transcript_reviewed','alignment_reviewed')) else uncertain).append(segment)
    return dict(recordings=len(profile['recordings']),inventory_minutes=sum(r['duration'] for r in profile['recordings'])/60,
                reviewed_minutes=sum(s['end']-s['start'] for s in approved)/60,uncertain_segments=len(uncertain),
                coverage_status='unmeasured until reviewed alignment and acoustic annotation',
                missing=['phonetic context','register','intensity','syllable density','sustained vowels','texture','register transitions'])
