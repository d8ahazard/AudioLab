from types import SimpleNamespace
from pathlib import Path
import json
import numpy as np
import pytest
from modules.separator.instrument_policy import MEGA_EXTRAS, selected, duplicate_of, INSTRUMENT_STEMS, DRUM_STEMS
from modules.separator.stem_separator import EnsembleDemucsMDXMusicSeparationModel as Pipeline
from modules.separator.stem_manifest import mixable_stems, write_json, MANIFEST_NAME


def test_mega_has_no_core_or_aggregate_competitors():
    assert not set(MEGA_EXTRAS) & {'drums','kick','snare','hh','toms','bass','double-bass',
        'guitar','acoustic-guitar','electric-guitar','piano','digital-piano','keys','strings',
        'bowed_strings','woodwind','wind','brass','vocal','lead-vocal','back-vocal','percussion','bells'}
    assert selected({},'mega_stems',MEGA_EXTRAS,[]) == []
    with pytest.raises(ValueError): selected({'mega_stems':['drums']},'mega_stems',MEGA_EXTRAS,[])


def test_duplicate_guard_does_not_confuse_unison_or_silence():
    t=np.arange(16000)/16000
    a=np.stack([np.sin(2*np.pi*220*t)]*2).astype('float32')
    assert duplicate_of(a, {'original':a.copy()}) == 'original'
    assert duplicate_of(a, {'quieter':a*.9}) is None
    assert duplicate_of(a, {'different':a+.01*np.sin(2*np.pi*440*t)}) is None
    assert duplicate_of(np.zeros_like(a), {'empty':np.zeros_like(a)}) is None


def test_consensus_uses_two_original_drum_parents():
    p=Pipeline.__new__(Pipeline)
    p.options={}; p._advance_progress=lambda _:None
    state={}
    p.separator=SimpleNamespace(load_model=lambda name:state.update(model=name))
    x=np.zeros((2,3000),dtype='float32')
    original=[]
    def separate(a,sr,**kw):
        if kw['task']=='instruments':
            scale=1 if state['model']=='htdemucs_6s.yaml' else 2
            result={k:x+scale*.01 for k in INSTRUMENT_STEMS+['vocals']}
            original.append(result['drums'])
            return result
        assert a is original[len(seen)]
        seen.append(a)
        return {'drums_'+k:a*.1 for k in DRUM_STEMS}
    p._separate_as_arrays_current=separate
    r={'song':{'instrumental':x,'mix_np':x,'sr':44100,'output_folder':'.'}}
    seen=[]
    p._multistem_separation_all(r)
    p._advanced_drum_separation_all(r)
    assert len(seen)==2 and set('drums_'+k for k in DRUM_STEMS)<=r['song'].keys()


def test_export_keeps_silent_selected_stems_and_respects_user_choices(tmp_path,monkeypatch):
    import modules.separator.stem_separator as module
    monkeypatch.setattr(module,'model_fingerprint',lambda *_:{})
    p=Pipeline.__new__(Pipeline)
    p.options={'vocal_reverb':'Keep wet','smart_stems':'conservative'}
    p.instrument_stems=['bass'];p.drum_stems=['kick'];p.smart_stems='conservative'
    p.separation_profile=SimpleNamespace(value='hybrid_cleaned')
    p.separator=SimpleNamespace(model_file_dir='.',loaded_models=set(),run_records=[])
    p._advance_progress=lambda _:None
    x=np.zeros((2,3000),dtype='float32')
    outputs=p._save_all_stems({'song':{'output_folder':str(tmp_path),'sr':44100,'mix_np':x,
        'bass':x,'guitar':x,'drums_kick':x,'drums_snare':x}})
    assert len(outputs)==2
    m=json.loads((tmp_path/MANIFEST_NAME).read_text())
    assert {e['role'] for e in m['stems']}=={'bass','drums_kick'}
    assert all(not e['hidden'] for e in m['stems'])


def test_mega_alternatives_and_duplicates_never_double_mix(tmp_path):
    entries=[{'filename':'other.wav','role':'other'}, {'filename':'flute.wav','role':'mega_flute'},
             {'filename':'copy.wav','role':'mega_oboe','duplicate_of':'mega_flute'}]
    write_json(tmp_path/MANIFEST_NAME,{'stems':entries})
    assert mixable_stems(['other.wav','flute.wav','copy.wav'],tmp_path)==['other.wav']
    assert mixable_stems(['flute.wav','copy.wav'],tmp_path)==['flute.wav']
    assert mixable_stems(['copy.wav'],tmp_path)==['copy.wav']
