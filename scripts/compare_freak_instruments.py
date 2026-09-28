"""Named instrument/specialist screening, isolated from application defaults."""
import argparse
import html
import json
import logging
import re
import sys
import tempfile
from functools import partial
from unittest.mock import patch
from importlib.metadata import version
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from modules.separator.model_runtime import AudioLabSeparator, CUSTOM_MODELS, download_verified
from modules.separator.stem_manifest import file_hash, model_fingerprint, write_json

OUT = REPO / 'outputs/separation_v4_validation/freak-instruments'
SOURCE = REPO / 'outputs/process/Freak OG_0a636df0/source/Freak OG.wav'
MEGA = 'mvsep_mega_model_bs_roformer_53_stems_v1.ckpt'
MEGA_CONFIG = 'mvsep_mega_model_bs_roformer_53_stems.yaml'
CORE = ['bs_roformer_vocals_resurrection_unwa.ckpt', 'melband_roformer_big_beta7.ckpt',
        'mel_band_roformer_instrumental_becruily.ckpt']
STARTS = [12, 65, 130, 210]
SR = 44100

CUSTOM_MODELS['becruily_guitar.ckpt'] = {
    'repo': 'becruily/mel-band-roformer-guitar', 'revision': '6409e7f88754b07ef7ca3bd1b76a15f010f1672a',
    'checkpoint': 'becruily_guitar.ckpt', 'config': 'config_guitar_becruily.yaml', 'license': 'unspecified',
    'checkpoint_sha256': '83472bbf125774af5282d2e0b86df89eaf2dd45e8a4ec8d68e820ebf3e42a83c',
    'config_sha256': 'b681c3f886251b04b666b3f06e87ce65d7ec610e40b5d75915c01782e5444b0e',
}


class ScreeningSeparator(AudioLabSeparator):
    def load_model(self, name, **kwargs):
        if name == MEGA:
            # The architecture supports arbitrary output heads, but the older
            # generic config validator caps them at 16. Only this pinned model
            # is allowed through with exactly 53; all other checks still run.
            from audio_separator.separator.roformer.parameter_validator import ParameterValidator
            with patch.dict(ParameterValidator.PARAMETER_RANGES, {'num_stems': (53, 53)}):
                return super().load_model(name, **kwargs)
        if name == 'becruily_guitar.ckpt':
            # 0.47.0's MelBand constructor accepts but drops this argument
            # when constructing MaskEstimator. Scope the fix to this process
            # and this one author-pinned model; never edit installed packages.
            from audio_separator.separator.uvr_lib_v5.roformer import mel_band_roformer as mel
            with patch.object(mel, 'MaskEstimator', partial(mel.MaskEstimator, mlp_expansion_factor=1)):
                return super().load_model(name, **kwargs)
        return super().load_model(name, **kwargs)

    def download_model_files(self, name):
        if name != MEGA:
            return super().download_model_files(name)
        base = 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.21/'
        for filename, digest in [(MEGA, 'c62820893bbf86d4e734f966bd142d9157cfc8bb8e79e9d8f9ea553f3ff3519f'),
                                 (MEGA_CONFIG, '7e198062a251587088adb91215a4f44ab59e67bd62fcc805cf54d6e7dfc51103')]:
            download_verified(base + filename, Path(self.model_file_dir) / filename, digest)
        self.model_is_uvr_vip = False
        self.model_friendly_name = name
        return name, 'MDXC', name, str(Path(self.model_file_dir) / name), MEGA_CONFIG


def read(path):
    a, sr = sf.read(path, dtype='float32', always_2d=True)
    assert sr == SR and a.shape[1] == 2 and np.isfinite(a).all(), str(path)
    return a


def write(path, a):
    assert a.ndim == 2 and a.shape[1] == 2 and np.isfinite(a).all(), str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, a, SR, subtype='FLOAT')


def run(separator, name, inputs, group):
    folder = OUT / 'predictions' / group
    cache = folder / 'complete.json'
    identity = {'revision': 'freak-screen-1', 'package': version('audio-separator'), 'model': name,
                'inputs': {p.name: file_hash(p) for p in inputs}, 'quality': 'balanced', 'preserve_gain': True}
    if name == 'becruily_guitar.ckpt':
        identity['adapter'] = 'honor-author-mask-mlp-expansion-1'
    if name == MEGA:
        identity['adapter'] = 'allow-author-53-head-count'
    if cache.exists():
        data = json.loads(cache.read_text())
        if data['identity'] == identity and data['models'] == model_fingerprint(separator.model_file_dir, [name]) and all(
                (folder / p).exists() and file_hash(folder / p) == h for p, h in data['outputs'].items()):
            print('Cached ' + group, flush=True)
            return
    print('Starting ' + group, flush=True)
    separator.load_model(name)
    folder.mkdir(parents=True, exist_ok=True)
    outputs, runs = {}, []
    for index, source in enumerate(inputs):
        expected = len(read(source))
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            separator.output_dir = temp
            separator.model_instance.output_dir = temp
            paths = separator.separate(str(source))
            labels = set()
            for p in paths:
                p = Path(p) if Path(p).is_absolute() else Path(temp) / p
                found = re.findall(r'\(([^()]*)\)', p.stem)
                if not found:
                    raise ValueError('Unlabelled output: ' + str(p))
                role = found[0].lower().replace(' ', '-')
                if role in labels or not re.fullmatch(r'[a-z0-9_-]+', role):
                    raise ValueError('Ambiguous output role: ' + role)
                labels.add(role)
                a = read(p)
                assert len(a) == expected, (p, len(a), expected)
                target = folder / str(index) / (role + '.wav')
                write(target, a)
                outputs[str(target.relative_to(folder))] = file_hash(target)
            expected_roles = {
                'htdemucs_6s.yaml': {'vocals', 'drums', 'bass', 'guitar', 'piano', 'other'},
                'BS-Roformer-SW.ckpt': {'vocals', 'drums', 'bass', 'guitar', 'piano', 'other'},
                'becruily_guitar.ckpt': {'guitar', 'other'},
                'MDX23C-DrumSep-aufr33-jarredou.ckpt': {'kick', 'snare', 'toms', 'hh', 'ride', 'crash'},
            }.get(name)
            if (expected_roles is not None and labels != expected_roles) or (name == MEGA and len(labels) != 53):
                raise ValueError(f'Missing/unexpected model outputs: {name}: {labels}')
            runs.append(separator.run_records[-1])
    write_json(cache, {'identity': identity, 'models': model_fingerprint(separator.model_file_dir, [name]),
                       'outputs': outputs, 'runs': runs})
    print('Completed ' + group, flush=True)


def prepare():
    a, _ = librosa.load(SOURCE, sr=SR, mono=False)
    write_json(OUT / 'source.json', {'source': str(SOURCE), 'sha256': file_hash(SOURCE),
        'excerpt_starts': STARTS, 'listen_seconds': 12, 'context_seconds_each_side': 4,
        'note': 'Each 20-second excerpt inferred independently; central 12 seconds auditioned.'})
    paths = []
    for i, start in enumerate(STARTS):
        p = OUT / 'inputs' / f'{i}.wav'
        write(p, a[:, (start-4)*SR:(start+16)*SR].T)
        paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--mega-only', action='store_true', help='Resume broad-stem pass from saved instrumental inputs')
    args = parser.parse_args()
    if args.report_only:
        report()
        return
    if args.mega_only:
        separator = ScreeningSeparator(model_file_dir=str(REPO / 'models/audio_separator'),
            use_soundfile=True, use_autocast=True, log_level=logging.ERROR, quality='balanced', preserve_gain=True)
        run(separator, MEGA, [OUT / 'instrumental' / f'{i}.wav' for i in range(4)], 'mega-instrumental')
        report()
        return
    inputs = prepare()
    separator = ScreeningSeparator(model_file_dir=str(REPO / 'models/audio_separator'),
        use_soundfile=True, use_autocast=True, log_level=logging.ERROR, quality='balanced', preserve_gain=True)
    for i, model in enumerate(CORE):
        run(separator, model, inputs, f'core-{i}')
    inst = []
    for i in range(len(inputs)):
        parts = []
        for k in range(3):
            folder = OUT / 'predictions' / f'core-{k}' / str(i)
            # Vocal-target RoFormers label their accompaniment "other".
            p = folder / 'instrumental.wav'
            if not p.exists():
                p = folder / 'other.wav'
            parts.append(read(p))
        p = OUT / 'instrumental' / f'{i}.wav'
        write(p, np.mean(parts, axis=0))
        inst.append(p)
    for key, model in [('demucs', 'htdemucs_6s.yaml'), ('sw', 'BS-Roformer-SW.ckpt')]:
        run(separator, model, inst, key + '-instrumental')
        run(separator, model, inputs, key + '-mix')
    run(separator, 'becruily_guitar.ckpt', inst, 'guitar-specialist')
    for key in ['demucs', 'sw']:
        drums = [OUT / 'predictions' / (key + '-instrumental') / str(i) / 'drums.wav' for i in range(4)]
        run(separator, 'MDX23C-DrumSep-aufr33-jarredou.ckpt', drums, 'drumsep-' + key)
    run(separator, MEGA, inst, 'mega-instrumental')
    report()


def report():
    groups = [('demucs-instrumental', 'Demucs 6 / new instrumental'), ('sw-instrumental', 'BS-RoFormer SW / new instrumental'),
              ('demucs-mix', 'Demucs 6 / original mix'), ('sw-mix', 'BS-RoFormer SW / original mix'),
              ('guitar-specialist', 'Becruily guitar / new instrumental'), ('drumsep-demucs', 'DrumSep / Demucs drums'),
              ('drumsep-sw', 'DrumSep / SW drums'), ('mega-instrumental', 'MVSep Mega 53 / new instrumental')]
    tracks = []
    def add(group, title, role, paths):
        a = np.concatenate([read(p)[4*SR:16*SR] for p in paths])
        target = OUT / 'listen' / group / (role + '.wav')
        write(target, a)
        tracks.append({'group': group, 'model': title, 'role': role, 'path': target.relative_to(OUT).as_posix(),
                       'rms_db': float(20*np.log10(max(1e-12, np.sqrt(np.mean(a.astype('float64')**2))))),
                       'peak': float(abs(a).max())})
    add('reference', 'Original mix', 'reference', [OUT / 'inputs' / f'{i}.wav' for i in range(4)])
    add('reference', 'New instrumental', 'instrumental', [OUT / 'instrumental' / f'{i}.wav' for i in range(4)])
    for group, title in groups:
        folder = OUT / 'predictions' / group
        if not (folder / 'complete.json').exists():
            continue
        for p in sorted((folder / '0').glob('*.wav')):
            role = 'remainder-after-guitar' if group == 'guitar-specialist' and p.stem == 'other' else p.stem
            add(group, title, role, [folder / str(i) / p.name for i in range(4)])
    # One common downward-only gain across every player, never normalize quiet noise upwards.
    gain = min(1., .98 / max(t['peak'] for t in tracks))
    for t in tracks:
        p = OUT / t['path']
        write(p, read(p)*gain)
        t['sha256'] = file_hash(p)
    evidence = {p.parent.name: json.loads(p.read_text()) for p in (OUT / 'predictions').glob('*/complete.json')}
    write_json(OUT / 'listening-manifest.json', {'tracks': tracks, 'common_gain': gain, 'starts': STARTS,
        'predictions': evidence, 'source': json.loads((OUT / 'source.json').read_text()),
        'custom_model_metadata': {'guitar': CUSTOM_MODELS['becruily_guitar.ckpt'],
            'mega': {'release': 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21',
                     'license': 'checkpoint license not specified in release notes'}},
        'note': 'Mega outputs overlap: never sum all 53. No stems auto-hidden; low energy does not prove absence.'})
    priority = ['reference', 'instrumental', 'drums', 'bass', 'guitar', 'piano', 'synth', 'keys',
                'other', 'kick', 'snare', 'hh', 'toms', 'ride', 'crash', 'percussion']
    roles = sorted(set(t['role'] for t in tracks), key=lambda r: (priority.index(r) if r in priority else 100, r))
    cards = []
    for role in roles:
        label = {'hh': 'Hi-hat (hh)', 'bowed_strings': 'Bowed strings'}.get(role, role.replace('-', ' ').title())
        cards.append('<section><h2>' + html.escape(label) + '</h2>')
        for t in tracks:
            if t['role'] == role:
                cards.append(f'<article><strong>{html.escape(t["model"])}</strong><small>{t["rms_db"]:.1f} dBFS · original level</small>'
                    f'<audio controls preload="none" src="{t["path"]}"></audio></article>')
        choices = ''.join(f'<option>{html.escape(t["model"])}</option>' for t in tracks if t['role'] == role)
        cards.append(f'<label>Winner <select data-role="{role}"><option>Unrated</option><option>Tie</option>{choices}</select></label>'
                     f'<textarea data-role="{role}" placeholder="Notes"></textarea></section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Freak — instrument workshop</title>
<style>body{background:#111820;color:#e9eef5;font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px}h1{font-size:36px}p{line-height:1.6;color:#b8c7d6}section{border-top:1px solid #34485d;padding:20px 0}article{display:inline-block;vertical-align:top;background:#1d2a38;padding:16px;margin:5px;border-radius:10px;width:310px}audio{width:100%;margin-top:12px}small{display:block;color:#96aec3;margin-top:8px}select,textarea,button{background:#24394a;color:white;border:1px solid #527084;border-radius:5px;padding:10px}textarea{display:block;width:95%;margin-top:12px}a{color:#8bcfff}nav{position:sticky;top:0;background:#111820;padding:14px 0;z-index:2}</style>
<h1>Huxlxy — Freak: instrument workshop</h1>
<p>Named comparisons on Freak OG.wav. Four 12-second excerpts: 0:12, 1:05, 2:10, 3:30. Each was processed separately with four seconds of context on either side. Player times 0, 12, 24 and 36 begin the excerpts.</p>
<p>Start with drums, bass, guitar, piano and synth. DrumSep splits isolated drums; Mega explores additional instruments. Mega families and children overlap—these are alternatives, not 53 tracks to sum. Quiet outputs stay available for listening; no noise gating or upward normalization.</p>
<nav><label>Find stem <input id="filter" placeholder="drums, kick, synth…"></label> <button id="export">Save winner / notes</button></nav>
<p><a href="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21">Mega author release</a> · <a href="https://huggingface.co/becruily/mel-band-roformer-guitar">Guitar model</a> · <a href="listening-manifest.json">Measurements and file hashes</a></p>
''' + ''.join(cards) + '''<script>
const key='freak-instruments-1';let scores=JSON.parse(localStorage.getItem(key)||'{}');
document.querySelectorAll('[data-role]').forEach(e=>{const f=e.tagName==='SELECT'?'winner':'notes';e.value=scores[e.dataset.role]?.[f]|| (f==='winner'?'Unrated':'');e.onchange=()=>{(scores[e.dataset.role]??={})[f]=e.value;localStorage.setItem(key,JSON.stringify(scores));};});
document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause();})));
document.querySelector('#filter').oninput=e=>document.querySelectorAll('section').forEach(s=>s.hidden=!s.querySelector('h2').textContent.toLowerCase().includes(e.target.value.toLowerCase()));
document.querySelector('#export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({schema:1,comparison:key,scores},null,2)],{type:'application/json'}));a.download='freak-instrument-scores.json';a.click();URL.revokeObjectURL(a.href);};
</script></html>'''
    (OUT / 'index.html').write_text(page, encoding='utf-8')
    print(f'Listening page: {OUT / "index.html"}; {len(tracks)} players; gain {gain}', flush=True)


if __name__ == '__main__':
    main()
