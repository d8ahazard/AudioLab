"""Freak ensemble auditions, orchestral Mega screening and Smart Stems review."""
import argparse
import html
import json
import logging
import sys
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
import compare_freak_instruments as engine
from modules.separator.audio_quality import consensus_blend, activity_review
from modules.separator.stem_manifest import file_hash, write_json

ROOT = REPO / 'outputs/separation_v4_validation'
BASE = ROOT / 'freak-instruments'
SR = 44100


def read(p):
    a,sr=sf.read(p,dtype='float32',always_2d=True)
    assert sr==SR and a.shape[1]==2 and np.isfinite(a).all()
    return a


def write(p,a):
    p.parent.mkdir(parents=True,exist_ok=True)
    assert np.isfinite(a).all()
    sf.write(p,a,SR,subtype='FLOAT')


def blend_group(out, prefix, first, second, count, roles):
    for i in range(count):
        a={r:read(first/str(i)/(r+'.wav')) for r in roles}
        b={r:read(second/str(i)/(r+'.wav')) for r in roles}
        avg={r:(a[r]+b[r])*.5 for r in roles}
        for r in roles:
            others=sum(v for k,v in avg.items() if k!=r)
            clean=consensus_blend(a[r].T,b[r].T,others.T).T
            write(out/'predictions'/(prefix+'-average')/str(i)/(r+'.wav'),avg[r])
            write(out/'predictions'/(prefix+'-cleaned')/str(i)/(r+'.wav'),clean)


def montage(paths):
    return np.concatenate([read(p)[4*SR:16*SR] for p in paths])


def page(out,title,description,entries,starts,extra=None):
    """Each entry has role/model/audio/parent. Analyze before playback gain."""
    out.mkdir(parents=True,exist_ok=True)
    tracks=[]
    gain=min(1.,.98/max(float(abs(e['audio']).max()) for e in entries))
    for n,e in enumerate(entries):
        activity=activity_review(e['audio'].T,e['parent'].T,SR)
        relative=Path('.hidden_stems' if activity['hidden'] else 'listen')/f'{n:03d}-{e["role"]}.wav'
        write(out/relative,e['audio']*gain)
        tracks.append({'role':e['role'],'model':e['model'],'path':relative.as_posix(),
                       'sha256':file_hash(out/relative),'activity':activity})
    counts={k:sum(t['activity']['display_status']==k for t in tracks) for k in ['keep','review','hidden']}
    manifest={'revision':'instrument-ensemble-1','starts':starts,'seconds_each_excerpt':12,
              'common_gain':gain,'tracks':tracks,'counts':counts,'metadata':extra or {},
              'note':'Review is not an absence decision. No destructive deletion; all files retained.'}
    write_json(out/'manifest.json',manifest)
    priority=['reference','instrumental','drums','bass','guitar','piano','synth','other','vocals','kick','snare','hh','toms','ride','crash','reconstruction','unassigned']
    roles=sorted({t['role'] for t in tracks},key=lambda r:(priority.index(r) if r in priority else 100,r))
    cards=[]
    for role in roles:
        group=[t for t in tracks if t['role']==role]
        status='keep' if any(t['activity']['display_status']=='keep' for t in group) else 'review'
        cards.append(f'<section data-status="{status}"><h2>{html.escape(role.replace("_"," ").replace("-"," ").title())}</h2>')
        for t in group:
            a=t['activity']
            reason=a['review_reason'] or a['reason']
            cards.append(f'<article><b>{html.escape(t["model"])}</b><small>{a["display_status"].upper()} · {a["relative_db"]:.1f} dB relative to parent<br>{html.escape(reason)}</small><audio preload="none" controls src="{t["path"]}"></audio></article>')
        options=''.join('<option>'+html.escape(t['model'])+'</option>' for t in group)
        cards.append(f'<p><label>Winner <select data-role="{role}"><option>Unrated</option><option>Tie</option><option>No useful stem</option>{options}</select></label></p><textarea data-role="{role}" placeholder="Notes"></textarea></section>')
    content='''<!doctype html><html lang="en"><meta charset="utf-8"><title>''' + html.escape(title) + '''</title>
<style>body{background:#101820;color:#e8f0f7;font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 24px}p{line-height:1.6;color:#b7cadb}h1{font-size:36px}section{padding:20px 0;border-top:1px solid #385165}article{display:inline-block;vertical-align:top;width:310px;padding:16px;background:#1d2d3a;margin:5px;border-radius:9px}small{display:block;color:#a9bdd0;margin-top:9px}audio{width:100%;margin-top:12px}textarea{display:block;width:95%;margin:10px 0}select,textarea,button,input{background:#243b4e;color:white;border:1px solid #58768e;padding:9px;border-radius:5px}nav{background:#101820;position:sticky;top:0;padding:15px 0;z-index:2}a{color:#81cdfa}</style>
<h1>''' + html.escape(title) + '</h1><p>'+description+'</p><p>Excerpt starts: '+', '.join(f'{int(s)//60}:{int(s)%60:02}' for s in starts)+'''. Each player joins the central 12 seconds of each independently processed excerpt.</p>
<p>Smart Stems keeps clear activity visible. Quiet/uncertain parts are placed in review, not declared absent. Only digital silence or very faint stationary broadband noise qualifies for automatic hiding. Tonal or transient content vetoes noise hiding. All files remain available below; no upward normalization.</p>
<nav><input id="filter" placeholder="Find instrument…"> <label><input type="checkbox" id="review"> Show quiet / review stems</label> <button id="export">Save winner + notes</button></nav>
<p>''' + f'{counts["keep"]} active estimates · {counts["review"]} review · {counts["hidden"]} silence/noise. ' + '<a href="manifest.json">Measurements and source metadata</a></p>' + ''.join(cards) + '''<script>
const key=location.pathname;let scores=JSON.parse(localStorage.getItem(key)||'{}');
document.querySelectorAll('[data-role]').forEach(e=>{let f=e.tagName==='SELECT'?'winner':'notes';e.value=scores[e.dataset.role]?.[f]||(f==='winner'?'Unrated':'');e.onchange=()=>{(scores[e.dataset.role]??={})[f]=e.value;localStorage.setItem(key,JSON.stringify(scores));};});
function filter(){let q=document.querySelector('#filter').value.toLowerCase();document.querySelectorAll('section').forEach(s=>s.hidden=!s.querySelector('h2').textContent.toLowerCase().includes(q)||(!q&&!document.querySelector('#review').checked&&s.dataset.status!=='keep'));}
document.querySelector('#filter').oninput=filter;document.querySelector('#review').onchange=filter;filter();
document.querySelectorAll('audio').forEach(a=>a.onplay=()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause();}));
document.querySelector('#export').onclick=()=>{let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({schema:1,comparison:key,scores},null,2)],{type:'application/json'}));a.download='instrument-notes.json';a.click();URL.revokeObjectURL(a.href);};
</script></html>'''
    (out/'index.html').write_text(content,encoding='utf-8')
    print(title,counts,'players',len(tracks),flush=True)


def freak():
    out=ROOT/'freak-ensembles'
    roles=['drums','bass','guitar','piano','other','vocals']
    kit=['kick','snare','hh','toms','ride','crash']
    blend_group(out,'instruments',BASE/'predictions/demucs-instrumental',BASE/'predictions/sw-instrumental',4,roles)
    blend_group(out,'drums',BASE/'predictions/drumsep-demucs',BASE/'predictions/drumsep-sw',4,kit)
    engine.OUT=out
    sep=engine.ScreeningSeparator(model_file_dir=str(REPO/'models/audio_separator'),use_soundfile=True,
        use_autocast=True,quality='balanced',preserve_gain=True,log_level=logging.ERROR)
    engine.run(sep,'MDX23C-DrumSep-aufr33-jarredou.ckpt',
        [out/'predictions/instruments-cleaned'/str(i)/'drums.wav' for i in range(4)],'drumsep-clean-parent')
    inst=montage([BASE/'instrumental'/f'{i}.wav' for i in range(4)])
    entries=[dict(role='instrumental',model='New instrumental reference',audio=inst,parent=inst)]
    groups=[(BASE/'predictions/demucs-instrumental','Demucs / instrumental'),(BASE/'predictions/sw-instrumental','SW / instrumental'),
            (out/'predictions/instruments-average','50/50 blend'),(out/'predictions/instruments-cleaned','Consensus + gentle cleanup')]
    for folder,label in groups:
        pool={r:montage([folder/str(i)/(r+'.wav') for i in range(4)]) for r in roles}
        for r,a in pool.items():entries.append(dict(role=r,model=label,audio=a,parent=inst))
        reconstruction=sum(a for r,a in pool.items() if r!='vocals')
        entries.append(dict(role='reconstruction',model=label+' / five instrument stems summed',audio=reconstruction,parent=inst))
        entries.append(dict(role='unassigned',model=label+' / instrumental minus five stems',audio=inst-reconstruction,parent=inst))
    parent=montage([out/'predictions/instruments-cleaned'/str(i)/'drums.wav' for i in range(4)])
    for folder,label in [(BASE/'predictions/drumsep-demucs','DrumSep / Demucs'),(BASE/'predictions/drumsep-sw','DrumSep / SW'),
        (out/'predictions/drums-average','Blend DrumSep outputs 50/50'),(out/'predictions/drums-cleaned','Blend outputs + gentle cleanup'),
        (out/'predictions/drumsep-clean-parent','DrumSep on cleaned blended drums')]:
        for r in kit:
            entries.append(dict(role=r,model=label,audio=montage([folder/str(i)/(r+'.wav') for i in range(4)]),parent=parent))
    provenance={str(p.relative_to(ROOT)):file_hash(p) for p in (BASE/'predictions').glob('*/complete.json')}
    page(out,'Freak — instrument and drum ensembles',
        'Original candidates versus equal averaging and a stereo-linked consensus blend. Cleanup is capped at 3 dB and only acts on model disagreement where another stem dominates. This is a listening experiment, not a promoted recipe. The sixth model output is a vocal/residual bucket; it does not replace our approved lead vocals. Reconstructed mixes and unassigned remainders expose missing content; the remainder is never automatically added back.',
        entries,engine.STARTS,{'baseline_manifests':provenance,'algorithm':'consensus_blend max_cut_db=3; avg_wave weights=[0.5,0.5]',
                              'drum_run':json.loads((out/'predictions/drumsep-clean-parent/complete.json').read_text())})


def orchestra():
    out=ROOT/'orchestral-mega'
    source=out/'russian-easter.flac'
    a,_=librosa.load(source,sr=SR,mono=False)
    duration=a.shape[-1]/SR
    starts=[int(duration*f) for f in [.03,.17,.34,.52,.73,.92]]
    inputs=[]
    for i,start in enumerate(starts):
        p=out/'inputs'/f'{i}.wav';write(p,a[:,(start-4)*SR:(start+16)*SR].T);inputs.append(p)
    del a
    engine.OUT=out
    sep=engine.ScreeningSeparator(model_file_dir=str(REPO/'models/audio_separator'),use_soundfile=True,
        use_autocast=True,quality='balanced',preserve_gain=True,log_level=logging.ERROR)
    engine.run(sep,engine.MEGA,inputs,'mega-orchestra')
    reference=montage(inputs)
    entries=[dict(role='reference',model='Musopen orchestral recording',audio=reference,parent=reference)]
    folder=out/'predictions/mega-orchestra'
    for p in sorted((folder/'0').glob('*.wav')):
        entries.append(dict(role=p.stem,model='MVSep Mega 53',audio=montage([folder/str(i)/p.name for i in range(6)]),parent=reference))
    metadata=json.loads((out/'recording.json').read_text())
    metadata['run']=json.loads((folder/'complete.json').read_text())
    page(out,'Russian Easter — Mega 53 orchestral test',
        'Rimsky-Korsakov: Russian Easter Festival Overture, Op. 36. Recording: Musopen Kickstarter public-domain collection, via Internet Archive. '
        '<a href="https://archive.org/details/MusopenCollectionAsFlac">Recording and public-domain metadata</a> · '
        '<a href="https://www.bso.org/works/rimsky-korsakov-russian-easter-festival-overture">Orchestration</a>. '
        'This score covers strings, woodwinds, brass, harp, glockenspiel, timpani and other percussion. It does not contain all 53 model categories. Family and child estimates overlap: audition them as alternatives. A model label is not evidence the instrument is present. Mega remains opt-in and outside the normal separation recipe.',entries,starts,metadata)


def smart_audit():
    out=ROOT/'smart-stems-review'
    entries=[]
    parent=montage([BASE/'instrumental'/f'{i}.wav' for i in range(4)])
    folder=BASE/'predictions/mega-instrumental'
    for p in sorted((folder/'0').glob('*.wav')):
        entries.append(dict(role=p.stem,model='Freak / Mega 53',audio=montage([folder/str(i)/p.name for i in range(4)]),parent=parent))
    page(out,'Smart Stems — Freak audit','This view applies the proposed conservative visibility rules to the earlier Mega outputs. Use No useful stem and notes to identify musical bleed or false instrument labels. Those cases need instrument-aware validation; a volume threshold cannot safely decide them.',entries,engine.STARTS)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freak','orchestra','smart']);args=parser.parse_args()
    {'freak':freak,'orchestra':orchestra,'smart':smart_audit}[args.stage]()
