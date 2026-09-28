"""Versioned, dependency-light contracts for performer jobs and their timeline."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import json
import math
import re

SCHEMA = 1


class FitError(ValueError):
    """Requested text cannot be realized within its phrase budget."""


def words(text):
    return re.findall(r"[\w]+(?:['’][\w]+)*", text.casefold())


def identity(path):
    path=Path(path); digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return dict(path=str(path.resolve()),sha256=digest.hexdigest(),bytes=path.stat().st_size)


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)


@dataclass
class PhraseEdit:
    start: float
    end: float
    text: str

    def validate(self, duration):
        if not all(math.isfinite(x) for x in (self.start,self.end)) or not 0<=self.start<self.end<=duration:
            raise ValueError('Edit must have a finite, positive window within the source timeline')
        if not words(self.text):raise ValueError('Edit requires desired words')


@dataclass
class PerformerRequest:
    source_audio: str
    performer_profile: str
    output_dir: str
    delivery_mode: str = 'performer'
    source_transcript: str | None = None
    target_lyrics: str | None = None
    phrase_edits: list = field(default_factory=list)
    delivery_strength: float = 1.
    seed: int = 20260924
    stage: str = 'm1_guide'
    guide_conditioning: str = 'transcript'

    def validate(self):
        for path in (self.source_audio,self.performer_profile):
            if not Path(path).is_file():raise FileNotFoundError(path)
        if self.delivery_mode not in ('preserve','performer'):raise ValueError('Unknown delivery mode')
        if self.stage not in ('m1_guide','learned'):raise ValueError('Unknown performer stage')
        if self.guide_conditioning not in ('transcript','audio_only'):raise ValueError('Unknown guide conditioning')
        if not isinstance(self.seed,int) or not 0<=self.seed<2**32:raise ValueError('Seed must be an integer in [0, 2**32)')
        if not math.isfinite(self.delivery_strength) or not 0<=self.delivery_strength<=1:
            raise ValueError('delivery_strength must be within [0,1]')
        if self.target_lyrics and self.phrase_edits:raise ValueError('Use full target lyrics OR phrase edits')
        if self.delivery_mode=='preserve' or self.delivery_strength==0:
            if self.phrase_edits or (self.target_lyrics and (not self.source_transcript or words(self.target_lyrics)!=words(self.source_transcript))):
                raise ValueError('Preserve mode cannot change words; use performer mode')
        if self.stage=='m1_guide' and self.delivery_mode=='performer' and self.delivery_strength not in (0.,1.):
            raise ValueError('M1 is an untrained control and supports strength 0 or 1 only')
        return self


def validate_edits(edits,duration):
    parsed=sorted([PhraseEdit(**e) if isinstance(e,dict) else e for e in edits],key=lambda e:e.start)
    previous=0.
    for edit in parsed:
        edit.validate(duration)
        if edit.start<previous:raise ValueError('Edit windows overlap')
        previous=edit.end
    return parsed


def allocate_frames(weights, minimums, total, maximums=None):
    """Water-fill an integer budget; reject infeasible edits instead of clipping words."""
    import numpy as np
    if not isinstance(total,(int,np.integer)) or total<1:raise ValueError('Frame budget must be a positive integer')
    w=np.asarray(weights,dtype=float)
    bounds=[np.asarray(minimums,dtype=float),np.asarray(maximums if maximums is not None else [total]*len(w),dtype=float)]
    if any(not np.isfinite(b).all() or np.any(b!=np.floor(b)) for b in bounds):raise ValueError('Bounds must be finite integer frames')
    lo,hi=[b.astype(int) for b in bounds]
    if w.ndim!=1 or not len(w) or w.shape!=lo.shape or hi.shape!=w.shape or not np.isfinite(w).all() or np.any(w<=0) or np.any(lo<1) or np.any(hi<lo):
        raise ValueError('Invalid duration constraints')
    if not int(lo.sum())<=total<=int(hi.sum()):raise FitError(f'Phrase requires {lo.sum()}–{hi.sum()} frames; budget is {total}')
    out=lo.copy();remaining=total-int(out.sum())
    while remaining:
        active=out<hi; shares=w*active;shares=shares/shares.sum()*remaining
        add=np.minimum(np.floor(shares).astype(int),hi-out)
        if not add.any():add[np.argmax(np.where(active,shares,-1))]=1
        out+=add;remaining-=int(add.sum())
    return out
