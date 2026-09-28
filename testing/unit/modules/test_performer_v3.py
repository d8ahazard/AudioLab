import json
import numpy as np
import pytest
from modules.rvc_v3.performer.contracts import allocate_frames, FitError, validate_edits, PerformerRequest
from modules.rvc_v3.performer.render import guide_time_map
from modules.rvc_v3.performer.analysis import phrase_windows


def test_duration_budgets_and_caps():
    rng=np.random.default_rng(4)
    for _ in range(100):
        lo=rng.integers(1,10,20);hi=lo+rng.integers(0,30,20)
        total=int(rng.integers(lo.sum(),hi.sum()+1))
        result=allocate_frames(rng.random(20)+.01,lo,total,hi)
        assert result.sum()==total
        assert np.all(result>=lo) and np.all(result<=hi)
    with pytest.raises(FitError):allocate_frames([1,1],[8,8],10)
    with pytest.raises(ValueError):allocate_frames([1],[1],2.5)
    with pytest.raises(ValueError):allocate_frames([1],[float('nan')],10)


def test_edits_cannot_overlap_or_spill():
    with pytest.raises(ValueError):validate_edits([dict(start=0,end=4,text='a'),dict(start=3,end=5,text='b')],6)
    with pytest.raises(ValueError):validate_edits([dict(start=4,end=7,text='a')],6)


def test_guide_overfull_fails_without_truncation():
    with pytest.raises(FitError):guide_time_map(np.ones(100,dtype=bool),10)
    mapping,report=guide_time_map(np.r_[np.zeros(10),np.ones(100),np.zeros(10)],200)
    assert len(mapping)==200 and np.all(np.diff(mapping)>=0)
    assert sum(x['frames'] for x in report['runs'])==200


def test_preserve_is_direct_existing_pipeline(tmp_path,monkeypatch):
    from modules.rvc_v3.performer.contracts import identity
    import scripts.rvc_backend_worker as v2
    from scripts.performer_worker import run
    source=tmp_path/'source.wav';source.write_bytes(b'source')
    asset=tmp_path/'asset';asset.write_bytes(b'checkpoint')
    profile=tmp_path/'profile.json'
    profile.write_text(json.dumps({**{k:identity(asset) for k in ('checkpoint','index','reference')},'baseline_settings':{'index_rate':.5,'protect':.2}}))
    calls=[]
    def baseline(req):calls.append(req);return {'output':'unchanged.wav'}
    monkeypatch.setattr(v2,'run',baseline)
    req=PerformerRequest(str(source),str(profile),str(tmp_path/'out'),delivery_mode='preserve')
    assert run(req)['output']=='unchanged.wav'
    assert len(calls)==1 and calls[0]['backend']=='rvc_v2' and calls[0]['source']==str(source)
    req.target_lyrics='different words'
    with pytest.raises(ValueError):run(req)


def test_models_use_conditioning_and_backpropagate():
    import torch
    from modules.rvc_v3.performer.models import DeliveryPlanner,AcousticBridge
    torch.set_num_threads(2);torch.manual_seed(1)
    planner=DeliveryPlanner(20,2,width=32,layers=1,heads=4).eval()
    tokens=torch.tensor([[1,2,3,0]]);mask=tokens==0
    context=torch.zeros(1,4,8)
    a=planner(tokens,torch.tensor([0]),context,torch.tensor([0]),mask)
    b=planner(tokens,torch.tensor([1]),context,torch.tensor([0]),mask)
    assert not torch.allclose(a,b) and not a[0,-1].any()
    bridge=AcousticBridge(width=32,layers=1,heads=4)
    result=bridge(torch.randn(1,4,768),context,torch.randn(1,4,768),mask)
    assert result.shape==(1,4,768) and torch.isfinite(result).all()
    result.square().mean().backward()
    assert bridge.memory.weight.grad.abs().sum()>0
    with pytest.raises(ValueError):bridge(torch.randn(1,4,768),context,torch.randn(1,4,768),torch.ones_like(mask))


def test_phrase_windows_do_not_rewrite_words():
    from types import SimpleNamespace
    req=SimpleNamespace(source_transcript=None,phrase_edits=[],target_lyrics=None)
    observed={'text':'One two','words':[dict(start=1,end=2,text='One'),dict(start=2.1,end=3,text='two')]}
    result=phrase_windows(observed,4,req)
    assert result[0]['text']=='One two' and result[0]['end']<=4
    req.source_transcript='Three four'
    with pytest.raises(ValueError):phrase_windows(observed,4,req)


def test_performance_plan_preserves_phonetic_distinctions():
    from modules.rvc_v3.performer.plan import PerformancePlan,PhoneEvent,seconds_to_frame
    plan=PerformancePlan('a',1,2,'artist','singing',1,[PhoneEvent('a','ɑ',1,1.8,'vowel')])
    assert plan.to_dict()['phones'][0]['realized']=='ɑ'
    assert seconds_to_frame(1.23,48000,480)==123
    plan.phones[0].end=2.1
    with pytest.raises(FitError):plan.validate()


def test_empty_review_set_blocks_training(tmp_path):
    from modules.rvc_v3.performer.data import reviewed_training_segments
    with pytest.raises(ValueError,match='No reviewed'):reviewed_training_segments({'recordings':[]})


def test_cosyvoice3_prompt_boundary_in_both_modes():
    from scripts.performer_guide_worker import generate
    from unittest.mock import Mock
    model=Mock();req=dict(reference='ref.wav',reference_text='Reference words')
    generate(model,'Desired words',req)
    assert '<|endofprompt|>' in model.inference_zero_shot.call_args.args[1]
    req['conditioning']='audio_only'
    generate(model,'Desired words',req)
    assert model.inference_cross_lingual.call_args.args[0].endswith('<|endofprompt|>Desired words')


def test_corrected_alignment_handles_only_frame_rounding(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from modules.rvc_v3.performer.analysis import align_corrected
    audio=tmp_path/'source.wav';audio.write_bytes(b'fixture')
    aligned={'word':'hello','start':.1,'end':1.02,'score':.9}
    module=SimpleNamespace(load_align_model=lambda *a,**kw:(object(),{}),align=lambda *a,**kw:{'word_segments':[aligned]})
    monkeypatch.setitem(sys.modules,'whisperx.alignment',module)
    result=align_corrected(audio,'hello',1,device='cpu')
    assert result['words'][0]['end']==1 and result['words'][0]['raw_end']==1.02
    aligned['end']=1.1
    with pytest.raises(FitError):align_corrected(audio,'hello',1,device='cpu')
    aligned['end']=1;aligned['score']=.02
    with pytest.raises(FitError):align_corrected(audio,'hello',1,device='cpu')


def test_phrase_edit_does_not_change_unedited_audio():
    from modules.rvc_v3.performer.assembly import place_phrase
    rng=np.random.default_rng(1)
    original=rng.uniform(-.3,.3,48000).astype('float32');output=original.copy()
    replacement=np.full(4800,2,dtype='float32')
    report=place_phrase(output,10000,replacement,48000,edited=True)
    assert np.array_equal(output[:10000],original[:10000])
    assert np.array_equal(output[14800:],original[14800:])
    assert abs(output[10000:14800]).max()<=.98 and report['local_gain']<1
    assert output[10000]==original[10000] and output[14799]==original[14799]
    before=output.copy()
    with pytest.raises(FitError):place_phrase(output,47000,replacement,48000,edited=True)
    assert np.array_equal(output,before)


def test_proportional_timing_preserves_native_durations():
    from modules.rvc_v3.performer.timing import proportional_frames
    lengths=np.array([10,100,20,50,5])
    lo=np.array([7,45,14,23,4]);hi=np.array([15,800,30,400,8])
    assert np.array_equal(proportional_frames(lengths,lo,int(lengths.sum()),hi),lengths)
    voiced=np.r_[np.zeros(10),np.ones(100),np.zeros(20),np.ones(50),np.zeros(5)]
    mapping,_=guide_time_map(voiced,len(voiced))
    assert np.array_equal(mapping,np.arange(len(voiced)))
    for total in range(int(lo.sum()),int(hi.sum())+1,11):
        result=proportional_frames(lengths,lo,total,hi)
        assert result.sum()==total and np.all(result>=lo) and np.all(result<=hi)
