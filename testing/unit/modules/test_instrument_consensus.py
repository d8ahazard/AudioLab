import numpy as np
import pytest
from modules.separator.audio_quality import consensus_blend, activity_review, stereo


def test_consensus_preserves_agreement_and_stereo():
    t = np.arange(16000)/16000
    a = np.stack([.1*np.sin(2*np.pi*330*t), -.08*np.sin(2*np.pi*330*t)])
    np.testing.assert_allclose(consensus_blend(a, a, a*10), a, atol=1e-7)


def test_cancellation_guard_keeps_content():
    t = np.arange(16000)/16000
    a = stereo(.1*np.sin(2*np.pi*330*t))
    out = consensus_blend(a, -a)
    np.testing.assert_allclose(out, a, atol=1e-7)


def test_cleanup_reduces_disputed_bleed_without_deleting_common_tone():
    t = np.arange(48000)/16000
    target = stereo(.05*np.sin(2*np.pi*330*t))
    bleed = stereo(.02*np.sin(2*np.pi*1600*t))
    a, b = target + bleed, target
    plain = (a+b)*.5
    clean = consensus_blend(a, b, bleed*10)
    assert np.mean((clean-target)**2) < np.mean((plain-target)**2)
    assert np.mean(clean**2) > .9*np.mean(target**2)


@pytest.mark.parametrize('kind', ['tone', 'entrance', 'transient', 'noisy_instrument', 'anti_phase'])
def test_review_never_auto_hides_meaningful_sparse_content(kind):
    sr=16000
    t=np.arange(sr*3)/sr
    parent=stereo(.1*np.sin(2*np.pi*220*t))
    x=stereo(1e-5*np.sin(2*np.pi*660*t))
    if kind=='entrance':
        x[:,:sr]=0; x[:,sr+800:]=0
    elif kind=='transient':
        x[:]=0; x[:,sr]=.0001
    elif kind=='noisy_instrument':
        x=stereo(np.random.default_rng(9).normal(0,.0001,len(t)))
        x[:,:sr]=0; x[:,sr+1600:]=0
    elif kind=='anti_phase':
        x[1]*=-1
    assert not activity_review(x,parent,sr)['hidden']


def test_silence_hidden_and_mismatch_rejected():
    a=np.zeros((2,16000),dtype='float32')
    assert activity_review(a,a,16000)['hidden']
    with pytest.raises(ValueError):
        consensus_blend(a,a[:,:100])
