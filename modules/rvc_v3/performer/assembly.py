"""Timeline-safe phrase placement with local gain and edit-boundary blending."""
import numpy as np
from .contracts import FitError


def place_phrase(output,start,audio,sample_rate,edited=False):
    audio=np.asarray(audio,dtype=np.float32).copy()
    end=start+len(audio)
    if output.ndim!=1 or audio.ndim!=1 or not len(audio) or not np.isfinite(audio).all():raise ValueError('Invalid phrase waveform')
    if start<0 or end>len(output):raise FitError('Phrase exceeds source timeline')
    peak=float(abs(audio).max());gain=min(1.,.98/max(peak,1e-8));audio*=gain
    edge=min(round(.005*sample_rate),len(audio)//2)
    if edge:
        ramp=np.linspace(0,1,edge,dtype=np.float32)
        if edited:
            audio[:edge]=output[start:start+edge]*(1-ramp)+audio[:edge]*ramp
            audio[-edge:]=audio[-edge:]*(1-ramp)+output[end-edge:end]*ramp
        else:
            audio[:edge]*=ramp;audio[-edge:]*=ramp[::-1]
    output[start:end]=audio
    return dict(start_frame=start,end_frame=end,edge_frames=edge,local_gain=gain,peak_before_gain=peak)
