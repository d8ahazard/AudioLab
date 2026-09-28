"""Serializable performer plan, independent of waveform and model frame grids."""
from dataclasses import dataclass,field,asdict
import math
from .contracts import FitError


@dataclass
class PhoneEvent:
    canonical: str
    realized: str
    start: float
    end: float
    kind: str  # consonant, vowel, pause or breath
    stress: float = 0.
    energy: float = 0.
    texture: str | None = None


@dataclass
class PerformancePlan:
    desired_text: str
    phrase_start: float
    phrase_end: float
    performer_id: str
    delivery_mode: str
    seed: int
    phones: list[PhoneEvent] = field(default_factory=list)
    note_anchors: list[dict] = field(default_factory=list)
    reference_ids: list[str] = field(default_factory=list)
    schema: int = 1
    timing_unit: str = 'absolute_seconds'

    def validate(self):
        if self.delivery_mode not in ('rap','singing'):raise ValueError('Unknown delivery head')
        if not all(math.isfinite(t) for t in (self.phrase_start,self.phrase_end)) or not 0<=self.phrase_start<self.phrase_end:
            raise FitError('Invalid phrase window')
        previous=self.phrase_start
        for phone in self.phones:
            if phone.kind not in ('consonant','vowel','pause','breath'):raise ValueError('Unknown phone kind')
            if not all(math.isfinite(t) for t in (phone.start,phone.end,phone.stress,phone.energy)):
                raise ValueError('Nonfinite performance event')
            if not previous<=phone.start<phone.end<=self.phrase_end:raise FitError('Phone overlaps or exceeds phrase')
            if phone.kind in ('consonant','vowel') and (not phone.canonical or not phone.realized):raise ValueError('Keep canonical and realized phones separately')
            previous=phone.end
        for anchor in self.note_anchors:
            if not self.phrase_start<=anchor['time']<=self.phrase_end or not math.isfinite(anchor['midi']):
                raise ValueError('Invalid musical anchor')
        return self

    def to_dict(self):
        self.validate()
        return asdict(self)


def seconds_to_frame(seconds,sample_rate,hop):
    if not math.isfinite(seconds) or seconds<0 or sample_rate<=0 or hop<=0:raise ValueError('Invalid time grid')
    return round(seconds*sample_rate/hop)
