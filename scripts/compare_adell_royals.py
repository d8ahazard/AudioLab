"""Clone the approved wet Royals lead with Adell and audition captured reverb."""
import html
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from modules.rvc.configs.config import Config
from modules.rvc.infer.modules.vc.pipeline import VC
from modules.reverb_ir import restore_reverb
from modules.separator.stem_manifest import file_hash,write_json

ROOT=REPO/'outputs/separation_v4_validation'
OUT=ROOT/'adell-royals'
SONG='Lorde - Royals US Version_5365a577'
SOURCE=ROOT/'background-comparison/predictions'/SONG/'karaoke/vocals.wav'
IR=ROOT/'operational/reverb_modes/Capture_reverb/Royals__(Vocals).ir'


def stereo_at(path,sr=44100):
    x,rate=sf.read(path,dtype='float32',always_2d=True)
    assert np.isfinite(x).all()
    if rate!=sr:
        x=librosa.resample(x.T,orig_sr=rate,target_sr=sr).T
    if x.shape[1]==1:
        x=np.repeat(x,2,axis=1)
    assert x.shape[1]==2
    return x


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    settings=dict(model='Adell.pth',sid=0,f0_up_key=0,f0_method='rmvpe+',index_rate=1.,
        filter_radius=3,rms_mix_rate=.9,protect=.2,merge_type='median',crepe_hop_length=160,
        f0_autotune=False,rmvpe_onnx=False,clone_stereo=False,pitch_correction=False,
        pitch_correction_humanize=.95,use_model_warmup=True,warmup_duration=10.)
    identity={'source_sha256':file_hash(SOURCE),'model_sha256':file_hash(REPO/'models/trained/Adell.pth'),
        'index_sha256':file_hash(REPO/'models/trained/Adell.index'),'ir_sha256':file_hash(IR),'settings':settings}
    cache=OUT/'clone-complete.json'
    if cache.exists():
        cached=json.loads(cache.read_text())
        assert cached['identity']==identity
        clone=Path(cached['path'])
        assert file_hash(clone)==cached['sha256']
    else:
        vc=VC(Config(),True)
        vc.get_vc('Adell.pth')
        vc.index=str(REPO/'models/trained/Adell.index')
        outputs=vc.vc_multi(paths=[str(SOURCE)],project_dir=str(OUT),model_display_name='Adell',**settings)
        assert len(outputs)==1,outputs
        clone=Path(outputs[0])
        write_json(cache,{'identity':identity,'path':str(clone),'sha256':file_hash(clone)})
    original=stereo_at(SOURCE)
    before=stereo_at(clone)
    missing_frames=len(original)-len(before)
    if not 0 <= missing_frames <= round(.05*44100):
        raise ValueError(f'Unexpected RVC duration mismatch: {missing_frames} samples')
    # RVC frame rounding dropped 20 ms at the tail. Preserve its raw output and
    # append only bounded trailing silence for the aligned comparison copies.
    before=np.pad(before,((0,missing_frames),(0,0)))
    parts=[]
    with tempfile.TemporaryDirectory(dir=OUT) as temp:
        for start in range(0,len(before),8*44100):
            source_part=Path(temp)/'clone.wav'
            result_part=Path(temp)/'restored.wav'
            sf.write(source_part,before[start:start+8*44100],44100,subtype='FLOAT')
            restore_reverb(source_part,IR,result_part)
            parts.append(stereo_at(result_part))
    after=np.concatenate(parts)
    assert before.shape==after.shape==original.shape
    reapplied=OUT/'Adell-with-captured-IR.wav'
    sf.write(reapplied,after,44100,subtype='FLOAT')
    inst=stereo_at(ROOT/'hybrid-comparison/audio'/SONG/'instrumental.wav')
    bg=stereo_at(ROOT/'background-comparison/predictions'/SONG/'karaoke/bg_vocals.wav')
    assert inst.shape==bg.shape==original.shape
    tracks={'original-lead':original,'adell-wet':before,'adell-ir':after,
        'original-mix':original+inst+bg,'adell-mix':before+inst+bg,'adell-ir-mix':after+inst+bg}
    gain=min(1.,.98/max(float(abs(x).max()) for x in tracks.values()))
    exports={}
    for name,x in tracks.items():
        dest=OUT/(name+'.wav')
        sf.write(dest,x*gain,44100,subtype='FLOAT')
        check,rate=sf.read(dest,always_2d=True)
        assert check.shape==original.shape and rate==44100 and np.isfinite(check).all()
        exports[name]={'path':dest.name,'sha256':file_hash(dest),'peak_before_gain':float(abs(x).max())}
    write_json(OUT/'manifest.json',{'identity':identity,'source':str(SOURCE),'cloned_raw':str(clone),
        'ir':str(IR),'frames':len(original),'sample_rate':44100,'shared_playback_gain':gain,
        'clone_tail_padding_frames':missing_frames,'ir_context':'Reset between the three unrelated eight-second song excerpts',
        'input_condition':'Wet lead, no dereverb preprocessing; captured IR is added to the wet clone.',
        'clone_channels':'Mono RVC output; IR provides stereo effect. Original remains stereo.',
        'mix':'Same original instrumental and backing vocal in all three mixes; only the lead changes.', 'outputs':exports})
    columns=[]
    for title,vocal,mix in [('Original Royals lead','original-lead','original-mix'),
        ('Adell — wet clone','adell-wet','adell-mix'),('Adell — wet clone + captured IR','adell-ir','adell-ir-mix')]:
        columns.append(f'<section><h2>{html.escape(title)}</h2><h3>Lead vocal</h3><audio controls preload="none" src="{vocal}.wav"></audio><a download href="{vocal}.wav">Download vocal</a><h3>With instrumental + backing vocals</h3><audio controls preload="none" src="{mix}.wav"></audio><a download href="{mix}.wav">Download mix</a></section>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Adell × Royals — Reverb comparison</title><style>body{max-width:1250px;margin:36px auto;padding:20px;background:#131c29;color:#eff3f8;font:16px/1.5 system-ui}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}section{background:#223044;border-radius:12px;padding:22px}h2{font-size:20px}h3{font-size:16px}audio{width:100%}a{color:#a9d4ff}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style><h1>Adell × Royals</h1><p>The same 24-second sample: original lead, Adell clone, and Adell with the Sucial Fused captured response added.</p><p>The cloning input stayed wet. Pitch is unchanged. Each mix uses the same original instrumental and backing vocals. All six players share one gain to preserve differences; nothing is independently boosted.</p><div class="grid">COLUMNS</div><p>Voice conversion uses your local Adell model and index with RMVPE+ and the app’s normal mono-cloning setting. The captured IR adds stereo reflections. This is a voice-conversion comparison, not an original Adell performance. <a href="manifest.json">Settings and provenance</a></p><script>document.addEventListener('play',e=>{if(e.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true)</script></html>'''.replace('COLUMNS',''.join(columns))
    (OUT/'index.html').write_text(page,encoding='utf-8')
    print(OUT/'index.html',flush=True)


if __name__=='__main__':
    main()
