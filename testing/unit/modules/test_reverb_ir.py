import json
import numpy as np
import pytest
import soundfile as sf
from modules.reverb_ir import (fit_effect_ir, convolve_effect, capture_reverb,
    restore_reverb, save_capture, InvalidImpulseResponse)


def fixture_audio():
    sr = 2000
    rng = np.random.default_rng(13)
    dry = rng.normal(0, .08, (sr*9, 2))
    impulse = np.zeros((200, 2))
    impulse[29, 0], impulse[73, 1] = .3, .22
    impulse[120, 0], impulse[151, 1] = -.08, .05
    return sr, dry, impulse, convolve_effect(dry, impulse)


def test_known_stereo_ir_recovers_delay_and_generalizes():
    sr, dry, impulse, wet = fixture_audio()
    capture = fit_effect_ir(dry, wet, sr, max_ir_seconds=.1)
    assert capture['valid_for_restore']
    fitted = np.array(capture['impulse_response'])
    assert list(np.argmax(abs(fitted), axis=0)) == [29, 73]
    assert capture['diagnostics']['holdout_effect_nmse'] < .001
    unseen = np.random.default_rng(88).normal(0, .1, dry.shape)
    expected = convolve_effect(unseen, impulse)
    actual = convolve_effect(unseen, fitted)
    assert np.mean((actual-expected)**2) / np.mean(expected**2) < .001


def test_unrelated_removed_audio_fails_holdout():
    sr, dry, _, wet = fixture_audio()
    unrelated = np.random.default_rng(31).normal(0, .1, wet.shape)
    assert not fit_effect_ir(dry, unrelated, sr, max_ir_seconds=.1)['valid_for_restore']


def test_holdout_is_not_used_for_fitting():
    sr, dry, _, wet = fixture_audio()
    first = fit_effect_ir(dry, wet, sr, max_ir_seconds=.1)
    wet[-sr*3:] *= -1
    second = fit_effect_ir(dry, wet, sr, max_ir_seconds=.1)
    np.testing.assert_array_equal(first['impulse_response'], second['impulse_response'])
    assert not second['valid_for_restore']


def test_zero_effect_and_silent_excitation():
    sr, dry, _, wet = fixture_audio()
    capture = fit_effect_ir(dry, wet*0, sr, max_ir_seconds=.1)
    assert capture['valid_for_restore'] and capture['diagnostics']['no_effect']
    assert not fit_effect_ir(dry*0, wet, sr, max_ir_seconds=.1)['valid_for_restore']


def test_file_restore_keeps_gain_and_stereo_for_mono_clone(tmp_path):
    source = tmp_path/'mono.wav'
    sr = 2000
    mono = np.zeros(sr)
    mono[400] = .9
    sf.write(source, mono, sr, subtype='FLOAT')
    params = {'schema': 2, 'sample_rate': sr, 'valid_for_restore': True,
              'impulse_response': [[.5, .2], [0, .1]]}
    save_capture(tmp_path/'ir.json', params)
    out = restore_reverb(source, tmp_path/'ir.json', tmp_path/'out.wav')
    arr, rate = sf.read(out)
    assert rate == sr and arr.shape == (sr, 2)
    assert arr[400, 0] == pytest.approx(1.35, abs=1e-6)  # Float output, no clipping.
    assert arr[400, 1] == pytest.approx(1.08, abs=1e-6)
    assert arr[401, 1] == pytest.approx(.09, abs=1e-6)


def test_resample_ir_preserves_dc_gain_and_delay(tmp_path):
    source = tmp_path/'input.wav'
    sf.write(source, np.ones(16000)*.1, 4000, subtype='FLOAT')
    ir = np.zeros((200, 1)); ir[50] = .3
    save_capture(tmp_path/'ir.json', {'schema': 2, 'sample_rate': 2000, 'valid_for_restore': True, 'impulse_response': ir.tolist()})
    result = restore_reverb(source, tmp_path/'ir.json', tmp_path/'out.wav')
    arr, _ = sf.read(result)
    assert np.mean(arr[500:-500]) == pytest.approx(.13, abs=1e-4)
    assert abs(arr[10]-.1) < 1e-6


def test_rejects_legacy_and_failed_capture(tmp_path):
    for params in ({'impulse_response': [1]}, {'schema': 2, 'valid_for_restore': False, 'rejection_reasons': ['bad fit']}):
        save_capture(tmp_path/'ir.json', params)
        with pytest.raises(InvalidImpulseResponse):
            restore_reverb('unused.wav', tmp_path/'ir.json', tmp_path/'out.wav')


def test_short_capture_invalidates_previous_ir(tmp_path):
    for role in ('dry', 'wet'):
        sf.write(tmp_path/(role+'.wav'), np.ones(100), 2000, subtype='FLOAT')
    save_capture(tmp_path/'ir.json', {'old': True})
    capture_reverb(tmp_path/'dry.wav', tmp_path/'wet.wav', tmp_path/'ir.json')
    assert not json.loads((tmp_path/'ir.json').read_text())['valid_for_restore']


def test_capture_is_bound_to_source_and_hash(tmp_path):
    from modules.reverb_ir import capture_for_stem
    from modules.separator.stem_manifest import file_hash, MANIFEST_NAME
    capture = tmp_path/'song__(Vocals).ir'
    save_capture(capture, {'schema': 2})
    save_capture(tmp_path/MANIFEST_NAME, {'stems': [{'role': 'vocals', 'filename': 'song__(Vocals).wav',
        'reverb_ir': {'path': capture.name, 'sha256': file_hash(capture)}}]})
    assert capture_for_stem('song__(Vocals)(Cloned).wav', tmp_path) == str(capture)
    assert capture_for_stem('other__(Vocals).wav', tmp_path) is None
    capture.write_text('{}')
    with pytest.raises(InvalidImpulseResponse):
        capture_for_stem('song__(Vocals).wav', tmp_path)


def test_two_removal_stages_capture_total_effect_once(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    import modules.separator.stem_separator as pipeline_module
    pipeline = pipeline_module.EnsembleDemucsMDXMusicSeparationModel.__new__(pipeline_module.EnsembleDemucsMDXMusicSeparationModel)
    pipeline.reverb_removal = pipeline.echo_removal = 'Main Vocals'
    pipeline.crowd_removal = pipeline.noise_removal = 'Nothing'
    pipeline.delay_removal_model = 'echo'
    pipeline.crowd_removal_model = 'crowd'
    pipeline.noise_removal_model = 'noise'
    pipeline.store_reverb_ir = True
    pipeline.separator = MagicMock()
    pipeline._advance_progress = MagicMock()
    pipeline._separate_as_arrays_current = lambda audio, *a, **kw: {'dry': audio*.5}
    calls = []
    def capture(dry, wet, dest):
        calls.append((sf.read(dry)[0], sf.read(wet)[0]))
        save_capture(dest, {'valid_for_restore': True})
    monkeypatch.setattr(pipeline_module, 'extract_reverb', capture)
    source = np.ones((2, 200), dtype=np.float32)*.1
    result = pipeline._apply_transform_chain(source, 2000, 'song', 'vocals', str(tmp_path))
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0], (source*.25).T)
    np.testing.assert_allclose(calls[0][1], (source*.75).T)
    np.testing.assert_allclose(result, source*.25)


@pytest.mark.parametrize('mode,factor,capture_count', [('Keep wet',1.,0),('Dry vocals',.5,0),('Capture reverb',1.,1)])
def test_selected_fused_modes_keep_wet_unless_explicit_dry(monkeypatch, tmp_path, mode, factor, capture_count):
    from unittest.mock import MagicMock
    import modules.separator.stem_separator as pipeline_module
    from modules.separator.model_runtime import FUSED_DEREVERB
    pipeline = pipeline_module.EnsembleDemucsMDXMusicSeparationModel.__new__(pipeline_module.EnsembleDemucsMDXMusicSeparationModel)
    pipeline.options = {'vocal_reverb':mode}
    pipeline.reverb_removal = pipeline.echo_removal = 'Main Vocals'  # Saved legacy choices cannot silently dry Keep wet.
    pipeline.crowd_removal = pipeline.noise_removal = 'Nothing'
    pipeline.delay_removal_model = 'echo'
    pipeline.crowd_removal_model = 'crowd'
    pipeline.noise_removal_model = 'noise'
    pipeline.store_reverb_ir = False
    pipeline.separator = MagicMock()
    pipeline._advance_progress = MagicMock()
    pipeline._separate_as_arrays_current = lambda audio, *a, **kw: {'dry':audio*.5}
    calls=[]
    def capture(dry,wet,dest):
        calls.append((sf.read(dry)[0],sf.read(wet)[0]))
        save_capture(dest, {'schema':2,'valid_for_restore':False,'rejection_reasons':['heldout']})
    monkeypatch.setattr(pipeline_module,'extract_reverb',capture)
    source=np.ones((2,200),dtype=np.float32)*.1
    result=pipeline._apply_transform_chain(source,2000,'song','vocals',str(tmp_path))
    np.testing.assert_array_equal(result,source*factor)
    assert len(calls)==capture_count
    if mode!='Keep wet':
        pipeline.separator.load_model.assert_called_once_with(FUSED_DEREVERB)
    if capture_count:
        params=json.loads((tmp_path/'song__(Vocals).ir').read_text())
        assert params['exported_wet'] and params['restore_requested'] and params['cloning_input']=='wet'
        np.testing.assert_array_equal(calls[0][0],(source*.5).T)
        assert not list(tmp_path.glob('tmp*'))


def test_capture_mode_merge_skips_wet_original_but_restores_clone(tmp_path):
    from modules.reverb_ir import capture_for_stem
    from modules.separator.stem_manifest import file_hash,MANIFEST_NAME
    capture=tmp_path/'song__(Vocals).ir'
    save_capture(capture, {'schema':2,'exported_wet':True,'restore_requested':True,
        'valid_for_restore':False,'sample_rate':2000,'impulse_response':[[.1,.2]],
        'diagnostics':{'dry_rms':.1,'maximum_frequency_gain':.2}})
    save_capture(tmp_path/MANIFEST_NAME,{'stems':[{'role':'vocals','filename':'song__(Vocals).wav',
        'reverb_ir':{'path':capture.name,'sha256':file_hash(capture)}}]})
    assert capture_for_stem('song__(Vocals).wav',tmp_path) is None
    assert capture_for_stem('song__(Vocals)(Cloned).wav',tmp_path)==str(capture)
    assert capture_for_stem('song__(Vocals)(Cloned)(Re-Reverb).wav',tmp_path) is None
    clone=tmp_path/'clone.wav'
    sf.write(clone,np.ones((2000,2))*.1,2000,subtype='FLOAT')
    restore_reverb(clone,capture,tmp_path/'result.wav')
    result,_=sf.read(tmp_path/'result.wav')
    np.testing.assert_allclose(result,np.tile([.11,.12],(2000,1)),atol=1e-7)
