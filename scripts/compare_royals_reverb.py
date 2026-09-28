"""Named dereverb screening on the approved Royals main-vocal excerpt."""
import hashlib
import html
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from modules.separator.model_runtime import AudioLabSeparator, download_verified, read_outputs
from modules.separator.stem_manifest import file_hash, model_fingerprint, write_json

ROOT = REPO / 'outputs/separation_v4_validation'
OUT = ROOT / 'royals-reverb'
SOURCE = ROOT / 'background-comparison/predictions/Lorde - Royals US Version_5365a577/karaoke/vocals.wav'
CUSTOM = {
    'dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt': {
        'repo': 'anvuew/dereverb_bs_roformer', 'revision': 'bd5c6b55a429b4b74ce85fe5dcb690cfe36d91ec',
        'config': 'config.yaml', 'local_config': 'royals_dereverb_bs_anvuew_2026.yaml',
        'sha256': 'b4dfa4c4aa251c0afc303e5d512d904a1c2255e51d8d0dbfbddb18d740563d42', 'license': 'gpl-3.0'},
    'dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt': {
        'repo': 'Sucial/Dereverb-Echo_Mel_Band_Roformer', 'revision': '9c83bf27196213e6107cf0de5ff2d06d56a46876',
        'config': 'config_dereverb_echo_mbr_v2.yaml', 'local_config': 'royals_dereverb_echo_fused.yaml',
        'sha256': '1596b1063238f487d54a0510a8c92cb28c000c803a271dd618ac49efc99ef3f7', 'license': 'cc-by-nc-sa-4.0'},
}
MODELS = [
    ('anvuew_bs_2026', 'Anvuew BS-RoFormer — 2026', 'dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt', 'Newer stereo candidate; author repository updated February 13, 2026.'),
    ('sucial_fused', 'Sucial Dereverb + Echo — Fused', 'dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt', 'Author-recommended fusion for small and large reverb. Also targets delay; may remove vocal harmonies.'),
    ('anvuew_current', 'Anvuew Mel-Band — Current reverb model', 'dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt', 'Current AudioLab reverb-removal baseline.'),
    ('anvuew_gentle', 'Anvuew Mel-Band — Less aggressive', 'dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt', 'Gentler alternative: listen for preserved consonants and vocal body.'),
    ('sucial_v2', 'Sucial Dereverb + Echo — V2', 'dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt', 'Current AudioLab echo-removal baseline; also removes reverb.'),
    ('mdx23c', 'Aufr33 / Jarredou MDX23C — Legacy', 'MDX23C-De-Reverb-aufr33-jarredou.ckpt', 'Older architecture included as a reference.'),
]


class ScreeningSeparator(AudioLabSeparator):
    def download_model_files(self, model_filename):
        spec = CUSTOM.get(model_filename)
        if spec is None:
            return super().download_model_files(model_filename)
        base = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/"
        config = requests.get(base + spec['config'], timeout=45)
        config.raise_for_status()
        digest = hashlib.sha256(config.content).hexdigest()
        download_verified(base + spec['config'], Path(self.model_file_dir) / spec['local_config'], digest)
        download_verified(base + model_filename, Path(self.model_file_dir) / model_filename, spec['sha256'])
        self.model_is_uvr_vip = False
        self.model_friendly_name = model_filename
        return model_filename, 'MDXC', model_filename, str(Path(self.model_file_dir) / model_filename), spec['local_config']


def read(path):
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    assert rate == 44100 and audio.shape[1] == 2 and np.isfinite(audio).all(), path
    return audio


def write(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, 44100, subtype='FLOAT')
    np.testing.assert_array_equal(read(path), audio.astype(np.float32))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = read(SOURCE)
    sep = ScreeningSeparator(model_file_dir=str(REPO / 'models/audio_separator'), use_soundfile=True,
        use_autocast=True, preserve_gain=True, quality='balanced', log_level=logging.ERROR)
    records, errors, arrays = [], [], {'original': original}
    for slug, name, model, note in MODELS:
        folder = OUT / 'predictions' / slug
        identity = {'input_sha256': file_hash(SOURCE), 'model': model, 'revision': 'royals-reverb-1',
                    'custom': CUSTOM.get(model), 'quality': 'balanced', 'gain_preserved': True}
        cache = folder / 'complete.json'
        try:
            if cache.exists():
                record = json.loads(cache.read_text())
                assert record['identity'] == identity
                assert all(file_hash(folder / f'{role}.wav') == digest for role, digest in record['outputs'].items())
            else:
                folder.mkdir(parents=True, exist_ok=True)
                sep.load_model(model)
                with tempfile.TemporaryDirectory(dir=folder) as temp:
                    sep.output_dir = temp
                    sep.model_instance.output_dir = temp
                    outputs = sep.separate(str(SOURCE))
                    stems = read_outputs(outputs, temp, 44100, len(original), 'cleanup')
                    if 'dry' not in stems:
                        raise RuntimeError(f'Missing dry output: {outputs}; roles: {list(stems)}')
                    dry = stems['dry'].T
                    removed = original - dry
                    write(folder / 'dry.wav', dry)
                    write(folder / 'removed.wav', removed)
                    np.testing.assert_allclose(dry + removed, original, atol=1e-7)
                config_hash = None
                if model in CUSTOM:
                    config_hash = file_hash(Path(sep.model_file_dir) / CUSTOM[model]['local_config'])
                record = {'identity': identity, 'run': sep.run_records[-1],
                    'model_hashes': model_fingerprint(sep.model_file_dir, [model]), 'custom_config_sha256': config_hash,
                    'outputs': {r: file_hash(folder / f'{r}.wav') for r in ('dry', 'removed')}}
                write_json(cache, record)
            for role in ('dry', 'removed'):
                arr = read(folder / f'{role}.wav')
                assert arr.shape == original.shape
                arrays[slug + '-' + role] = arr
            records.append({'slug': slug, 'name': name, 'note': note, **record})
            print('Completed ' + name, flush=True)
        except Exception as error:
            errors.append({'model': model, 'name': name, 'error': repr(error)})
            write_json(OUT / 'errors.json', errors)
            print('FAILED ' + name + ': ' + repr(error), flush=True)
    if not records:
        raise RuntimeError('No dereverb candidate completed')
    # Same gain everywhere: do not disguise vocal loss or boost the removed signal.
    gain = min(1., .98 / max(1e-12, max(float(abs(a).max()) for a in arrays.values())))
    for name, audio in arrays.items():
        write(OUT / 'audio' / (name + '.wav'), audio * gain)
    write_json(OUT / 'manifest.json', {'source': str(SOURCE), 'input_sha256': file_hash(SOURCE),
        'frames': len(original), 'sample_rate': 44100, 'common_playback_gain': gain,
        'removed_signal': 'input minus predicted dry, not guaranteed pure reverb', 'models': records, 'errors': errors})
    cards = []
    for r in records:
        cards.append(f'<section><h2>{html.escape(r["name"])}</h2><p>{html.escape(r["note"])}</p><div class="players">'
            f'<div><h3>Treated / estimated dry</h3><audio controls preload="none" src="audio/{r["slug"]}-dry.wav"></audio><a download href="audio/{r["slug"]}-dry.wav">Download</a></div>'
            f'<div><h3>Removed signal</h3><audio controls preload="none" src="audio/{r["slug"]}-removed.wav"></audio><a download href="audio/{r["slug"]}-removed.wav">Download</a></div></div>'
            f'<label>Notes<textarea data-model="{r["slug"]}" placeholder="Tail removal, missing words, vocal texture..."></textarea></label></section>')
    options = '<option value="">Choose a preferred result</option><option>Original / no removal</option>' + ''.join(f'<option>{html.escape(r["name"])}</option>' for r in records)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Royals — Named reverb comparison</title>
<style>body{max-width:1080px;margin:36px auto;padding:20px;background:#121b27;color:#edf2f8;font:16px/1.5 system-ui}section{background:#202c3c;padding:22px;margin:22px 0;border-radius:12px}h2{font-size:21px}.players{display:grid;grid-template-columns:1fr 1fr;gap:24px}audio{width:100%}a{color:#a9d4ff}textarea{display:block;width:95%;min-height:55px}select,textarea,button{font:inherit;padding:9px}label{display:block;margin-top:18px}@media(max-width:650px){.players{grid-template-columns:1fr}}</style>
<h1>Royals — Reverb removal</h1><p>Named comparison on the extracted main vocal, after cleaned-hybrid separation and Frazer–Becruily backing-vocal removal. This is the same 24-second montage of three 8-second song excerpts.</p>
<section><h2>Original lead vocal</h2><audio controls preload="none" src="audio/original.wav"></audio><a download href="audio/original.wav">Download original</a></section>
<p>Listen for preserved words and vocal body in the treated track, and tails versus stolen vocal content in the removed signal. All tracks share one gain; removed signals are not boosted or assumed to be pure reverb. This is model screening, not an impulse-response reconstruction test.</p>
CARDS
<section><label>Preferred result <select id="winner">OPTIONS</select></label><button id="export">Export notes</button></section>
<p>Sources: <a href="https://huggingface.co/anvuew/dereverb_bs_roformer">Anvuew BS-RoFormer</a> · <a href="https://huggingface.co/anvuew/dereverb_mel_band_roformer">Anvuew Mel-Band</a> · <a href="https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer">Sucial models</a>. <a href="REPORT.md">Findings and IR pipeline review</a> · <a href="manifest.json">Run details</a></p>
<script>const store='royals-reverb-v1';let saved=JSON.parse(localStorage.getItem(store)||'{"notes":{}}');document.querySelector('#winner').value=saved.winner||'';document.querySelector('#winner').onchange=e=>{saved.winner=e.target.value;localStorage.setItem(store,JSON.stringify(saved))};document.querySelectorAll('textarea').forEach(t=>{t.value=saved.notes[t.dataset.model]||'';t.oninput=()=>{saved.notes[t.dataset.model]=t.value;localStorage.setItem(store,JSON.stringify(saved))}});document.querySelector('#export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({schema:1,comparison:'royals-reverb-v1',...saved},null,2)],{type:'application/json'}));a.download='royals-reverb-notes.json';a.click();URL.revokeObjectURL(a.href)};document.addEventListener('play',e=>{if(e.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true)</script></html>'''.replace('CARDS', '\n'.join(cards)).replace('OPTIONS', options)
    if errors:
        page = page.replace('<h1>', '<p>Some models failed; see errors.json for details.</p><h1>', 1)
    (OUT / 'index.html').write_text(page, encoding='utf-8')
    print(OUT / 'index.html', flush=True)


if __name__ == '__main__':
    main()
