"""Publish the listener-selected sample stems without rerunning or relabeling them."""
import html
import json
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from modules.separator.stem_manifest import file_hash, write_json

ROOT = REPO / 'outputs/separation_v4_validation'
OUT = ROOT / 'final-results'


def read(path):
    arr, sr = sf.read(path, dtype='float32', always_2d=True)
    if sr != 44100 or arr.shape[1] != 2 or not np.isfinite(arr).all():
        raise ValueError(f'Invalid cached audio: {path}')
    return arr


def main():
    key = json.loads((ROOT / 'background-comparison/key.json').read_text())
    excluded = next(p['song'] for p in key if p['id'] == 4)
    songs = list(dict.fromkeys(p['song'] for p in key if p['song'] != excluded))
    assert len(songs) == 5
    corpus = {p['name']: p for p in json.loads((ROOT / 'corpus.json').read_text())}
    cards, records = [], []
    labels = {'instrumental': 'Instrumental', 'vocals': 'Lead vocal', 'bg_vocals': 'Backing vocal', 'remerged': 'Re-merged'}
    for number, song in enumerate(songs, 1):
        predictions = ROOT / 'background-comparison/predictions' / song / 'karaoke'
        provenance = json.loads((predictions / 'complete.json').read_text())
        source_vocal = ROOT / 'hybrid-comparison/audio' / song / 'cleaned-vocals.wav'
        assert file_hash(source_vocal) == provenance['identity']['source_sha256']
        paths = {'instrumental': ROOT / 'hybrid-comparison/audio' / song / 'instrumental.wav',
                 'vocals': predictions / 'vocals.wav', 'bg_vocals': predictions / 'bg_vocals.wav'}
        for role in ('vocals', 'bg_vocals'):
            assert file_hash(paths[role]) == provenance['outputs'][role]
        tracks = {role: read(path) for role, path in paths.items()}
        assert len({arr.shape for arr in tracks.values()}) == 1
        tracks['remerged'] = tracks['instrumental'] + tracks['vocals'] + tracks['bg_vocals']
        # One gain across all four files preserves stem balance and avoids clipping.
        peak = max(float(abs(arr).max()) for arr in tracks.values())
        gain = min(1., .98 / max(peak, 1e-12))
        record = {'song': song, 'frames': len(tracks['vocals']), 'sample_rate': 44100,
                  'shared_export_gain': gain, 'pre_export_peak': peak,
                  'source_offsets_seconds': corpus[song]['excerpt_starts_seconds'],
                  'input_hashes': {role: {'path': str(path), 'sha256': file_hash(path)} for role, path in paths.items()},
                  'background_prediction': provenance, 'outputs': {}}
        title = song.rsplit('_', 1)[0]
        card = [f'<section><h2>{number}. {html.escape(title)}</h2><p class="meta">24-second sample · 44.1 kHz stereo</p><div class="tracks">']
        exports = {}
        for role, audio in tracks.items():
            dest = OUT / 'audio' / song / f'{role}.wav'
            dest.parent.mkdir(parents=True, exist_ok=True)
            exports[role] = (audio * gain).astype(np.float32)
            sf.write(dest, exports[role], 44100, subtype='FLOAT')
            np.testing.assert_array_equal(read(dest), exports[role])
            relative = dest.relative_to(OUT).as_posix()
            record['outputs'][role] = {'path': relative, 'sha256': file_hash(dest)}
            href = html.escape(quote(relative, safe='/'))
            card.append(f'<div><h3>{labels[role]}</h3><audio controls preload="none" src="{href}"></audio><a href="{href}" download="{html.escape(title)} - {labels[role]}.wav">Download WAV</a></div>')
        np.testing.assert_allclose(exports['instrumental'] + exports['vocals'] + exports['bg_vocals'], exports['remerged'], atol=2e-7, rtol=1e-6)
        card.append('</div></section>')
        cards.append('\n'.join(card))
        records.append(record)
    write_json(OUT / 'manifest.json', {'recipe': 'V4 instrumental + cleaned V2 vocals + Frazer–Becruily BS-RoFormer Karaoke',
        'selection': 'Explicit user selection after listening; not a universal quality claim',
        'excluded': {'background_pair': 4, 'song': excluded, 'reason': 'User reported no backing vocals'},
        'merge': 'instrumental + lead vocals + backing vocals; no aggregate double-counting',
        'source': 'Previously auditioned cached 24-second excerpt montages; not new full-song runs', 'songs': records})
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AudioLab — Final separation samples</title>
<style>body{max-width:1200px;margin:40px auto;padding:0 24px;background:#111823;color:#edf2f8;font:16px/1.5 system-ui}h1{font-size:32px;margin-bottom:6px}header{margin-bottom:30px}.tag{color:#9ee1c1;font-size:14px;letter-spacing:.08em}section{background:#1e2938;border:1px solid #314157;border-radius:14px;padding:24px;margin:22px 0}h2{font-size:21px;margin:0}h3{font-size:16px}p.meta,footer{color:#b7c5d6;font-size:14px}.tracks{display:grid;grid-template-columns:1fr 1fr;gap:22px 28px}audio{width:100%;display:block;margin-bottom:10px}a{color:#9ed2ff}footer{padding:16px 0 40px}@media(max-width:700px){.tracks{grid-template-columns:1fr}body{padding:0 14px}section{padding:18px}}</style></head><body>
<header><span class="tag">AUDIOLAB · SELECTED RECIPE</span><h1>Final separation samples</h1><p>V4 instrumental · Cleaned V2 vocals · Frazer–Becruily backing-vocal separation</p><p>Five samples, with the no-backing-vocal source from comparison #4 excluded. These are the same 24-second excerpts used for listening.</p></header>
CARDS
<footer>Re-merged = instrumental + lead vocal + backing vocal. All four tracks in each sample share one gain to preserve their balance. No individual upward normalization. <a href="manifest.json">Audio provenance</a></footer>
<script>document.addEventListener('play',event=>{if(event.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(audio=>{if(audio!==event.target)audio.pause()})},true)</script></body></html>'''.replace('CARDS', '\n'.join(cards))
    (OUT / 'index.html').write_text(page, encoding='utf-8')
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
