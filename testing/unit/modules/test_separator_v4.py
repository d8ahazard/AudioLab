import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from modules.separator.audio_quality import analyze_activity, fuse, stereo
from modules.separator.separation_profiles import selected_models, SeparationProfile
from modules.separator.stem_manifest import file_hash, write_json, hidden_stems, restore_stem, MANIFEST_NAME
from modules.separator.model_runtime import read_outputs, download_verified
from modules.separator.stem_separator import EnsembleDemucsMDXMusicSeparationModel as Pipeline


@pytest.fixture
def sounds():
    sr = 16000
    t = np.arange(sr * 3) / sr
    parent = stereo(.1 * np.sin(2 * np.pi * 220 * t))
    return sr, t, parent


def test_v4_has_three_equal_models_and_legacy_profiles_survive():
    specs = selected_models({"separation_profile": "v4"})
    assert len(specs) == 3
    assert [s.vocal_weight for s in specs] == [1, 1, 1]
    for profile, count in [("v1", 2), ("v2", 3), ("v3", 5)]:
        assert len(selected_models({"separation_profile": profile})) == count
    with pytest.raises(ValueError):
        selected_models({"separation_profile": "v4", "ensemble_size": 2})


def test_preset_routing_preserves_three_models_for_v4():
    assert len(selected_models({"separation_profile": "v2", "separation_preset": "remix"})) == 2
    assert len(selected_models({"separation_profile": "v4", "separation_preset": "remix"})) == 3


def test_default_hybrid_has_both_independent_model_sets():
    from wrappers.separate import Separate
    assert Separate.allowed_kwargs['separation_profile'].field.default == 'hybrid_cleaned'
    assert Separate.allowed_kwargs['backing_vocal_model'].field.default == 'karaoke'
    default = selected_models({})
    assert len(default) == 6
    assert [m.id for m in default] == [m.id for p in ('v2', 'v4') for m in selected_models({'separation_profile': p})]
    assert len(selected_models({'separation_profile': 'hybrid_cleaned', 'separation_preset': 'remix'})) == 6
    with pytest.raises(ValueError):
        selected_models({'separation_profile': 'hybrid_cleaned', 'ensemble_size': 3})


def test_hybrid_cleanup_matches_audition_and_preserves_stereo():
    from modules.separator.audio_quality import clean_hybrid_vocals
    from scripts.compare_hybrid_separation import clean_vocals
    rng = np.random.default_rng(21)
    vocal = rng.normal(0, .01, (2, 16000)).astype(np.float32)
    inst = vocal * 10
    mix = inst.copy()
    expected, _ = clean_vocals(vocal.T, (mix-inst).T, inst.T)
    result = clean_hybrid_vocals(vocal, mix, inst)
    np.testing.assert_allclose(result, expected.T, atol=1e-7)
    np.testing.assert_allclose(clean_hybrid_vocals(vocal, vocal+inst, inst), vocal, atol=1e-7)
    short = vocal[:, :100]
    np.testing.assert_array_equal(clean_hybrid_vocals(short, short, short*0), short)


def test_hybrid_routes_models_and_only_cleans_final_vocal(monkeypatch, tmp_path):
    from modules.separator.audio_quality import clean_hybrid_vocals
    model = Pipeline({'separation_profile': 'hybrid_cleaned', 'cpu': True})
    sample = np.ones((2, 4096), dtype=np.float32) * .1
    calls = []
    original = Pipeline._ensemble_separate_all
    def core(self, files):
        if self.separation_profile.value == 'hybrid_cleaned':
            return original(self, files)
        calls.append(self.separation_profile.value)
        self.separator.loaded_models.add(self.separation_profile.value)
        return {'song': {'vocals': sample * (1 if self.separation_profile.value == 'v2' else 3),
                         'instrumental': sample * (2 if self.separation_profile.value == 'v2' else 4),
                         'mix_np': sample * 5, 'sr': 44100}}
    monkeypatch.setattr(Pipeline, '_ensemble_separate_all', core)
    out = model._ensemble_separate_all([])['song']
    assert calls == ['v2', 'v4']
    np.testing.assert_array_equal(out['instrumental'], sample * 4)
    np.testing.assert_allclose(out['vocals'], clean_hybrid_vocals(sample, sample*5, sample*4))
    assert model.separator.loaded_models == {'v2', 'v4'}


@pytest.mark.parametrize("algorithm", ["avg_wave", "avg_complex", "median_magnitude", "min_magnitude", "max_magnitude"])
def test_fusion_preserves_phase_gain_and_length(sounds, algorithm):
    _, _, parent = sounds
    parent[1] *= -1
    out = fuse([parent, parent, parent], algorithm)
    np.testing.assert_allclose(out, parent, atol=1e-6)
    assert out.shape == parent.shape


def test_fusion_complements_reconstruct_known_mixture(sounds):
    _, t, vocals = sounds
    inst = stereo(.07 * np.sin(2 * np.pi * 503 * t))
    noise = stereo(np.random.default_rng(10).normal(0, .001, len(t)))
    v = fuse([vocals + noise, vocals - noise, vocals])
    i = fuse([inst - noise, inst + noise, inst])
    np.testing.assert_allclose(v + i, vocals + inst, atol=1e-7)


def test_silence_and_faint_stationary_hiss_are_hidden(sounds):
    sr, t, parent = sounds
    assert analyze_activity(np.zeros_like(parent), parent, sr).hidden
    hiss = np.random.default_rng(11).normal(0, 1e-5, parent.shape).astype(np.float32)
    assert analyze_activity(hiss, parent, sr).hidden
    assert not analyze_activity(hiss, parent, sr, noise_filter=False).hidden


@pytest.mark.parametrize("kind", ["quiet_tone", "brief_tone", "transient", "anti_phase", "musical_noise"])
def test_smart_stems_retains_sparse_or_quiet_content(sounds, kind):
    sr, t, parent = sounds
    quiet = stereo(1e-5 * np.sin(2 * np.pi * 330 * t))
    if kind == "brief_tone":
        quiet[:, :sr] = 0
        quiet[:, sr + 800:] = 0
    elif kind == "transient":
        quiet[:] = 0
        quiet[:, sr] = .0002
    elif kind == "anti_phase":
        quiet[1] *= -1
    elif kind == "musical_noise":
        quiet = np.random.default_rng(22).normal(0, 1e-4, parent.shape).astype(np.float32)
        quiet[:, :sr] = 0
        quiet[:, sr + 1600:] = 0
    assert not analyze_activity(quiet, parent, sr).hidden


def test_restore_verifies_hash_and_updates_cache(tmp_path):
    hidden = tmp_path / ".hidden_stems/song__(BG_Vocals).wav"
    hidden.parent.mkdir()
    sf.write(hidden, np.zeros((200, 2)), 44100, subtype="FLOAT")
    entry = {"filename": hidden.name, "path": str(hidden.relative_to(tmp_path)), "hidden": True, "sha256": file_hash(hidden)}
    write_json(tmp_path / MANIFEST_NAME, {"stems": [entry]})
    write_json(tmp_path / "separation_info.json", {"config": {}, "stems": []})
    target = restore_stem(tmp_path, entry["path"])
    assert Path(target).exists() and not hidden.exists()
    assert hidden_stems(tmp_path) == []
    assert json.loads((tmp_path / "separation_info.json").read_text())["stems"][0]["path"] == target
    with pytest.raises(ValueError):
        restore_stem(tmp_path, "../escape.wav")


def test_output_mapping_uses_stem_labels_not_model_name(tmp_path):
    for role in ("Vocals", "Other"):
        sf.write(tmp_path / f"tmp_({role})_instrumental_model.wav", np.ones((400, 2), dtype=np.float32) * .001, 16000)
    files = [p.name for p in tmp_path.glob("*.wav")]
    assert read_outputs(files, tmp_path, 16000, 400).keys() == {"vocals", "instrumental"}
    assert read_outputs(files, tmp_path, 16000, 400, "instruments").keys() == {"vocals", "other"}


@pytest.mark.parametrize("flag,stem,expected", [
    ("Main Vocals", "vocals", True), ("Main Vocals", "bg_vocals", False),
    ("All Vocals", "bg_vocals_2", True), ("All Vocals", "instrumental", False),
    ("All", "drums", True), ("Nothing", "vocals", False)])
def test_cleanup_targeting(flag, stem, expected):
    assert Pipeline._should_apply_transform(stem, flag) == expected


@pytest.mark.parametrize("setting,expected", [("echo_removal", "echo"), ("noise_removal", "noise"), ("crowd_removal", "crowd")])
def test_cleanup_can_run_independently(sounds, setting, expected):
    sr, _, parent = sounds
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.separator = MagicMock()
    pipeline.reverb_removal = pipeline.echo_removal = pipeline.noise_removal = pipeline.crowd_removal = "Nothing"
    setattr(pipeline, setting, "Main Vocals")
    pipeline.delay_removal_model, pipeline.noise_removal_model, pipeline.crowd_removal_model = "echo", "noise", "crowd"
    pipeline.store_reverb_ir = False
    pipeline._advance_progress = MagicMock()
    pipeline._separate_as_arrays_current = MagicMock(return_value={"dry": parent, "clean": parent})
    result = pipeline._apply_transform_chain(parent, sr, "song", "vocals", ".")
    pipeline.separator.load_model.assert_called_once_with(expected)
    np.testing.assert_array_equal(result, parent)


def test_missing_stem_is_an_error_not_silence(tmp_path, sounds):
    sr, _, parent = sounds
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.separator = MagicMock()
    pipeline.separator.separate.return_value = []
    with pytest.raises(RuntimeError, match="both primary stems"):
        pipeline._separate_as_arrays_current(parent, sr, output_folder=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_custom_download_rejects_corruption_without_network(tmp_path):
    p = tmp_path / "checkpoint"
    p.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_verified("https://unused.invalid", p, "0" * 64)


@pytest.mark.parametrize("quality,author,expected", [("fast", 4, 2), ("balanced", 4, 4), ("maximum", 4, 8), ("maximum", 12, 12)])
def test_quality_honors_author_defaults(monkeypatch, tmp_path, quality, author, expected):
    from types import SimpleNamespace
    from audio_separator.separator import Separator
    from modules.separator.model_runtime import AudioLabSeparator
    separator = AudioLabSeparator(cpu=True, info_only=True, quality=quality, model_file_dir=str(tmp_path))
    def load(self, filename, force_reload=False):
        settings = self.arch_specific_params["MDXC"]
        self.model_instance = SimpleNamespace(overlap=settings["overlap"] or author, batch_size=settings["batch_size"],
            override_model_segment_size=False, torch_device="cpu", effective_precision="fp32")
    monkeypatch.setattr(Separator, "load_model", load)
    separator.load_model("fake.ckpt")
    assert separator.model_instance.overlap == expected
    assert separator.model_instance.batch_size == 1
    assert separator.model_instance.normalization_threshold == float("inf")


def test_cpu_selection_is_instance_local(tmp_path):
    from modules.separator.model_runtime import AudioLabSeparator
    separator = AudioLabSeparator(cpu=True, info_only=True, model_file_dir=str(tmp_path))
    separator.setup_torch_device({})
    assert separator.torch_device.type == "cpu"
    assert separator.onnx_execution_provider == ["CPUExecutionProvider"]


def test_oom_is_reported_without_silent_quality_change(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import torch
    from audio_separator.separator import Separator
    from modules.separator.model_runtime import AudioLabSeparator
    separator = AudioLabSeparator(cpu=True, info_only=True, model_file_dir=str(tmp_path))
    separator.current_model = "fake"
    separator.model_instance = SimpleNamespace(torch_device=torch.device("cpu"), effective_precision="fp32", overlap=8)
    def oom(*a, **k):
        raise torch.cuda.OutOfMemoryError("test")
    monkeypatch.setattr(Separator, "separate", oom)
    with pytest.raises(RuntimeError, match="retry on CPU"):
        separator.separate("unused.wav")
    assert separator.model_instance.overlap == 8
    assert separator.run_records[-1]["status"] == "failed"


def test_aggregate_and_component_stems_are_not_mixed_twice(tmp_path):
    from modules.separator.stem_manifest import mixable_stems
    roles = ["vocals", "vocals_full", "bg_vocals", "bg_vocals_1", "instrumental", "drums", "drums_kick", "bass"]
    write_json(tmp_path / MANIFEST_NAME, {"stems": [{"role": r, "filename": r + ".wav"} for r in roles]})
    assert mixable_stems([r + ".wav" for r in roles], tmp_path) == ["vocals.wav", "bg_vocals.wav", "instrumental.wav"]
    assert mixable_stems(["vocals_(Cloned).wav", "vocals_full.wav", "instrumental.wav"], tmp_path) == ["vocals_(Cloned).wav", "instrumental.wav"]


def test_empty_smart_result_never_reprocesses_original_mix(tmp_path):
    from types import SimpleNamespace
    from wrappers.base_wrapper import BaseWrapper
    project = SimpleNamespace(last_outputs=[], output_dict={"stems": []}, project_dir=str(tmp_path), src_file="original.wav")
    assert BaseWrapper.filter_inputs(project) == ([], [])
    project.output_dict = {}
    write_json(tmp_path / "stems" / MANIFEST_NAME, {"stems": [{"hidden": True}]})
    assert BaseWrapper.filter_inputs(project) == ([], [])


def test_wrapper_cache_and_hidden_files_survive_cleanup(tmp_path, monkeypatch):
    from wrappers.separate import Separate
    import wrappers.separate as wrapper
    from types import SimpleNamespace
    source = tmp_path / "song.wav"
    sf.write(source, np.zeros((100, 2)), 44100, subtype="FLOAT")
    folder = tmp_path / "stems"
    folder.mkdir()
    visible = folder / "song__(Instrumental).wav"
    hidden = folder / ".hidden_stems/song__(BG_Vocals).wav"
    hidden.parent.mkdir()
    calls = []
    def separate(*args, **kwargs):
        calls.append(kwargs)
        for path in (visible, hidden):
            sf.write(path, np.zeros((100, 2)), 44100, subtype="FLOAT")
        write_json(folder / MANIFEST_NAME, {"stems": [
            {"path": str(p.relative_to(folder)), "filename": p.name, "hidden": p == hidden, "sha256": file_hash(p)}
            for p in (visible, hidden)]})
        return [str(visible)]
    monkeypatch.setattr(wrapper, "separate_music", separate)
    monkeypatch.setattr(Separate, "_cache_identity", staticmethod(lambda *args: {"fixed": "identity"}))
    project = SimpleNamespace(src_file=str(source), project_dir=str(tmp_path), file_dict={})
    project.add_output = lambda key, outputs: project.file_dict.update({key: outputs})
    sep = Separate()
    sep.process_audio([project], separation_profile="v4")
    sep.process_audio([project], separation_profile="v4")
    assert len(calls) == 1
    assert hidden.exists()
    assert project.file_dict["stems"] == [str(visible)]
    sep.process_audio([project], separation_profile="v4", separation_quality="maximum")
    assert len(calls) == 2


def test_same_basename_different_folders_does_not_collide(tmp_path, sounds):
    sr, _, parent = sounds
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.separator = MagicMock()
    pipeline.profile_models = selected_models({"separation_profile": "v4"})
    pipeline.separation_profile = SeparationProfile.V4_EXPERIMENTAL
    pipeline.ensemble_strength = 3
    pipeline.residual_blend_pct = 0
    pipeline.options = {}
    pipeline._advance_progress = MagicMock()
    pipeline._separate_as_arrays_current = MagicMock(return_value={"vocals": parent, "instrumental": parent})
    files = [{"base_name": "same", "key": str(tmp_path / str(i)), "mix_np": parent,
              "sr": sr, "output_folder": str(tmp_path / str(i))} for i in range(2)]
    results = pipeline._ensemble_separate_all(files)
    assert len(results) == 2
    assert all(r["base_name"] == "same" for r in results.values())
