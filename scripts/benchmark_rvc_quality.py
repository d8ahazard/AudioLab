"""Resumable RVC quality study. Originals are read-only; artifacts live on D:."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ROOT = Path('D:/AI-outputs/AudioLab/rvc_quality_validation')
OUTPUTS = Path('D:/AI-outputs/AudioLab')
MODELS = Path('D:/models/AudioLab/trained')
SETTINGS = dict(separation_profile='hybrid_cleaned', separation_quality='maximum',
                separate_bg_vocals=True, backing_vocal_model='karaoke', bg_vocal_layers=1,
                vocals_only=True, smart_stems='off', reverb_removal='Nothing',
                echo_removal='Nothing', noise_removal='Nothing', crowd_removal='Nothing')


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temp.replace(path)


def prepare(root):
    projects = OUTPUTS / 'process'
    specs = [
        ('hux_test', 'test', 'huxlxy', 'Huxlxy - Preacher Man Sloppy AI Remix_71ff2927'),
        ('shine_test', 'test', 'shinedown', 'Shinedown - Cracks in my Calmv2 Cover 1_10f2b696'),
        ('shine_devil', 'train', 'shinedown', 'Shinedown - DEVIL - Lyrics_1d98c2bb'),
        ('shine_cut_the_cord', 'train', 'shinedown', 'Shinedown - Cut The Cord Official Video_ec866968'),
        ('shine_simple_man', 'validation', 'shinedown', 'Shinedown - Simple Man Official Video_e52f7183'),
        ('shine_state_of_my_head', 'train', 'shinedown', 'Shinedown - State Of My Head Official Video_2bc4e500'),
    ]
    corpus = []
    for name, split, voice, project in specs:
        folder = projects / project
        sources = list((folder / 'source').glob('*.mp3'))
        if len(sources) != 1:
            raise ValueError(f'Expected one original mix: {folder}')
        old = list((folder / 'stems').glob('*__(Vocals).wav'))
        corpus.append(dict(id=name, split=split, voice=voice, source=fingerprint(sources[0]),
                           old_lead=fingerprint(old[0]) if len(old) == 1 else None))
    for name, filename in [
        ('monsters', 'Shinedown - MONSTERS (Official Video).mp3'),
        ('second_chance', 'Shinedown - Second Chance (Official Video) [HD].mp3'),
        ('sound_of_madness', 'Shinedown - Sound Of Madness (Official Video) [HD].mp3'),
    ]:
        corpus.append(dict(id='shine_' + name, split='train', voice='shinedown',
                           source=fingerprint(OUTPUTS / 'downloaded' / filename), old_lead=None))
    original_hux = [fingerprint(p) for p in sorted((OUTPUTS/'voices/huxlxy_v2/raw').glob('*.wav'))]
    manifest = dict(schema=1, seed=20260924, separation=SETTINGS, corpus=corpus,
                    hux_originals=original_hux,
                    models={name: fingerprint(MODELS / file) for name, file in {
                        'hux_latest':'huxlxy_v2_v500.pth', 'hux_secondary':'huxlxy_v2.pth',
                        'hux_index':'huxlxy_v2_v500.index', 'shine_index':'Shinedown666.index'}.items()},
                    shine_checkpoint=fingerprint(OUTPUTS/'voices/Shinedown666/saves/G_e75.pth'))
    write_json(root/'corpus.json', manifest)
    print('Prepared', root/'corpus.json', flush=True)


def export_shine(root):
    import torch
    from modules.rvc.infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid
    source = OUTPUTS/'voices/Shinedown666/saves/G_e75.pth'
    config = json.loads((source.parent.parent/'config.json').read_text())
    c, d = config['model'], config['data']
    args = [d['filter_length']//2+1, 32] + [c[k] for k in (
        'inter_channels','hidden_channels','filter_channels','n_heads','n_layers','kernel_size',
        'p_dropout','resblock','resblock_kernel_sizes','resblock_dilation_sizes','upsample_rates',
        'upsample_initial_channel','upsample_kernel_sizes','spk_embed_dim','gin_channels')] + [d['sampling_rate']]
    ckpt = torch.load(source, map_location='cpu', weights_only=True)
    weight = {k: v.half() for k,v in ckpt['model'].items() if not k.startswith('enc_q.')}
    model = SynthesizerTrnMs768NSFsid(*args, is_half=False)
    del model.enc_q
    model.load_state_dict(weight, strict=True)
    out = root/'models/Shinedown666_e75.pth'
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(weight=weight, config=args, info='75epoch', sr='48k', f0=1, version='v2'), out)
    import shutil
    shutil.copy2(MODELS/'Shinedown666.index', out.with_suffix('.index'))
    write_json(out.with_suffix('.json'), dict(source=fingerprint(source), output=fingerprint(out),
                                            validation='strict inference state dict load passed'))
    print('Exported', out, flush=True)


def separate(root, only=None):
    from modules.separator.stem_separator import separate_music
    manifest = json.loads((root/'corpus.json').read_text())
    for item in manifest['corpus']:
        if only and item['id'] not in only:
            continue
        folder = root/'separation'/item['id']
        complete = folder/'complete.json'
        if complete.exists():
            old = json.loads(complete.read_text())
            if old['source'] == fingerprint(item['source']['path']) and old['settings'] == SETTINGS:
                if all(fingerprint(p['path']) == p for p in old['outputs']):
                    print('Verified cache', item['id'], flush=True)
                    continue
        folder.mkdir(parents=True, exist_ok=True)
        print('Separating', item['id'], flush=True)
        start = time.perf_counter()
        paths = separate_music({str(folder): [item['source']['path']]}, **SETTINGS)
        if not paths:
            raise RuntimeError(f'No separation output for {item["id"]}')
        write_json(complete, dict(source=fingerprint(item['source']['path']), settings=SETTINGS,
                                 seconds=time.perf_counter()-start,
                                 outputs=[fingerprint(p) for p in paths]))
        print('Completed', item['id'], flush=True)


def excerpts(root):
    """Freeze energy-screened passages, with exact offsets for human review."""
    import numpy as np
    import soundfile as sf
    manifest=json.loads((root/'corpus.json').read_text())
    cases=[]
    for item in manifest['corpus']:
        if item['split'] != 'test': continue
        complete_path=root/'separation'/item['id']/'complete.json'
        if not complete_path.exists(): continue
        complete=json.loads(complete_path.read_text())
        lead=next(p['path'] for p in complete['outputs'] if p['path'].endswith('__(Vocals).wav'))
        audio,sr=sf.read(lead,dtype='float32',always_2d=True)
        seconds=len(audio)/sr
        # One active passage per quarter. No inference that energy implies a vocal technique.
        for i in range(4):
            candidates=np.arange(i*seconds/4,min((i+1)*seconds/4,seconds-12),2)
            if not len(candidates): continue
            start=float(max(candidates,key=lambda s: np.mean(audio[int(s*sr):int((s+12)*sr)]**2)))
            clip=root/'excerpts'/f'{item["id"]}_{i}.wav'
            clip.parent.mkdir(parents=True,exist_ok=True)
            samples=audio[int(start*sr):int((start+12)*sr)].mean(axis=1)
            # Float WAV headers may contain a timestamp. Preserve byte identity on resume.
            same=False
            if clip.exists():
                prior,prior_sr=sf.read(clip,dtype='float32')
                same=prior_sr==sr and np.array_equal(prior,samples)
            if not same: sf.write(clip,samples,sr,subtype='FLOAT')
            cases.append(dict(id=clip.stem,voice=item['voice'],source=fingerprint(clip),
                              parent=fingerprint(lead),start=start,seconds=12,
                              annotation='energy-screened; vocal-technique labels require listening'))
    write_json(root/'excerpts.json',cases)


def completed_job(result_path, request):
    """Resume only a matching request whose recorded inputs and output still match."""
    if not result_path.exists(): return False
    result=json.loads(result_path.read_text())
    if result.get('request')!=request: return False
    identities=list(result.get('inputs',{}).values())+[result.get('output_identity',{})]
    if not identities or any(not x.get('path') or not x.get('sha256') for x in identities): return False
    return all(Path(x['path']).is_file() and fingerprint(x['path'])['sha256']==x['sha256'] for x in identities)


def v2_jobs(root, only=None):
    import subprocess
    cases=json.loads((root/'excerpts.json').read_text())
    voices={'huxlxy':(MODELS/'huxlxy_v2_v500.pth',MODELS/'huxlxy_v2_v500.index'),
            'shinedown':(root/'models/Shinedown666_e75.pth',root/'models/Shinedown666_e75.index')}
    for case in cases:
        if only and not any(case['id'].startswith(x) for x in only): continue
        model,index=voices[case['voice']]
        # Full retrieval sweep; protection/pitch comparisons at a fixed central retrieval value.
        settings=[(r,.2,'rmvpe+') for r in (0.,.25,.5,.75,1.)]
        settings += [(.5,p,'rmvpe+') for p in (0.,.33,.5)]
        settings += [(.5,.2,p) for p in ('rmvpe','crepe','fcpe')]
        for rate,protect,pitch in settings:
            folder=root/'runs'/case['id']/f'v2_r{rate}_p{protect}_{pitch}'
            request=dict(backend='rvc_v2',checkpoint=str(model),index=str(index),
                         source=case['source']['path'],output_dir=str(folder),
                         index_rate=rate,protect=protect,pitch_method=pitch,seed=20260924,mode='preserve')
            result=folder/'result.json'
            if completed_job(result,request): continue
            write_json(folder/'request.json',request)
            with (folder/'run.log').open('w') as log:
                subprocess.run([sys.executable,str(REPO/'scripts/rvc_backend_worker.py'),str(folder/'request.json')],
                               cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
            print('Converted',case['id'],folder.name,flush=True)


def references(root):
    import numpy as np
    import soundfile as sf
    sources={'huxlxy': OUTPUTS/'voices/huxlxy_v2/raw/malevolent_vox.wav'}
    done=root/'separation/shine_devil/complete.json'
    if done.exists():
        sources['shinedown']=Path(next(p['path'] for p in json.loads(done.read_text())['outputs']
                                     if p['path'].endswith('__(Vocals).wav')))
    refs={}
    for voice,source in sources.items():
        audio,sr=sf.read(source,dtype='float32',always_2d=True)
        start=max(range(0,max(1,int(len(audio)/sr)-15),3),
                  key=lambda s: np.mean(audio[s*sr:(s+15)*sr]**2))
        out=root/'references'/f'{voice}.wav'
        out.parent.mkdir(parents=True,exist_ok=True)
        samples=audio[start*sr:(start+15)*sr].mean(axis=1)
        same=False
        if out.exists():
            prior,prior_sr=sf.read(out,dtype='float32');same=prior_sr==sr and np.array_equal(prior,samples)
        if not same: sf.write(out,samples,sr,subtype='FLOAT')
        refs[voice]=dict(source=fingerprint(source),start=start,audio=fingerprint(out),
                         selection='15 seconds, maximum mean energy among 3-second-spaced windows')
    write_json(root/'references.json',refs)


def external_jobs(root, backend, only=None, mode='preserve'):
    import subprocess
    from huggingface_hub import hf_hub_download
    refs=json.loads((root/'references.json').read_text())
    for case in json.loads((root/'excerpts.json').read_text()):
        if only and not any(case['id'].startswith(x) for x in only): continue
        if case['voice'] not in refs: continue
        folder=root/'runs'/case['id']/(backend if mode=='preserve' else backend+'_'+mode)
        request=dict(backend=backend,source=case['source']['path'],
                     reference=refs[case['voice']]['audio']['path'],output_dir=str(folder),
                     mode=mode,steps=50,seed=20260924)
        if mode=='target_style':
            transcripts=json.loads((root/'transcripts.json').read_text())
            request.update(lyrics=transcripts[case['id']]['text'],
                           reference_text=transcripts[case['voice']+'_reference']['text'],
                           transcript_reviewed=False)
        if backend == 'seed_vc':
            request['repository']='E:/dev/AudioLab-experiments/seed-vc'
            request['checkpoint']=hf_hub_download('Plachta/Seed-VC','DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth')
            request['config']=hf_hub_download('Plachta/Seed-VC','config_dit_mel_seed_uvit_whisper_base_f0_44k.yml')
        elif backend == 'yingmusic':
            request['repository']='E:/dev/AudioLab-experiments/YingMusic-SVC'
            request['checkpoint']=hf_hub_download('GiantAILab/YingMusic-SVC','YingMusic-SVC-full.pt')
            request['config']=request['repository']+'/configs/YingMusic-SVC.yml'
        else:
            request['repository']='E:/dev/AudioLab-experiments/Amphion'
            request['steps']=32
        result=folder/'result.json'
        if completed_job(result,request): continue
        write_json(folder/'request.json',request)
        with (folder/'run.log').open('w') as log:
            subprocess.run(['D:/venvs/audiolab-svc/Scripts/python.exe','-u',
                            str(REPO/'scripts/rvc_backend_worker.py'),str(folder/'request.json')],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Converted',case['id'],backend,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','export','separate','excerpts','v2',
                                         'references','seed_vc','yingmusic','vevo2','vevo2_style'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.root)
    elif args.action == 'export': export_shine(args.root)
    elif args.action == 'separate': separate(args.root, args.only)
    elif args.action == 'excerpts': excerpts(args.root)
    elif args.action == 'v2': v2_jobs(args.root,args.only)
    elif args.action == 'references': references(args.root)
    elif args.action=='vevo2_style': external_jobs(args.root,'vevo2',args.only,'target_style')
    else: external_jobs(args.root,args.action,args.only)


if __name__ == '__main__':
    main()
