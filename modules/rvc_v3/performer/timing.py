"""Bounded proportional durations: weights describe total time, not extra time."""
import numpy as np
from .contracts import FitError


def proportional_frames(weights,minimums,total,maximums):
    w=np.asarray(weights,dtype=float);lo=np.asarray(minimums,dtype=int);hi=np.asarray(maximums,dtype=int)
    if w.ndim!=1 or not len(w) or w.shape!=lo.shape or hi.shape!=w.shape or not np.isfinite(w).all() or np.any(w<=0) or np.any(lo<1) or np.any(hi<lo):
        raise ValueError('Invalid proportional duration constraints')
    if not isinstance(total,(int,np.integer)) or not lo.sum()<=total<=hi.sum():raise FitError('Infeasible duration budget')
    left=0.;right=float(np.max(hi/w))
    if not np.isfinite(right):raise ValueError('Duration weights have unsupported numeric range')
    for _ in range(80):
        middle=(left+right)/2
        if np.clip(middle*w,lo,hi).sum()<=total:left=middle
        else:right=middle
    desired=np.clip(left*w,lo,hi);out=np.floor(desired).astype(int)
    remaining=total-int(out.sum())
    while remaining:
        order=np.argsort(-(desired-out),kind='stable')
        for i in order:
            if out[i]<hi[i]:out[i]+=1;remaining-=1
            if remaining==0:break
    return out
