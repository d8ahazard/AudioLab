"""Separate dereverb audition copies; never replace the untreated training separation."""
import json
import logging
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_rvc_quality import ROOT, fingerprint, write_json

def main():
    import soundfile as sf
    import numpy as np
    from modules.separator.model_runtime import AudioLabSeparator,FUSED_DEREVERB,read_outputs
    engine=AudioLabSeparator(log_level=logging.ERROR,model_file_dir='D:/models/AudioLab/audio_separator',
        use_autocast=True,use_soundfile=True,normalization_threshold=1.,amplification_threshold=0.,
        quality='maximum',preserve_gain=True)
    engine.load_model(FUSED_DEREVERB)
    for item in json.loads((ROOT/'corpus.json').read_text())['corpus']:
        if item['voice']!='shinedown' or item['split']=='test': continue
        done=ROOT/'separation'/item['id']/'complete.json'
        if not done.exists(): continue
        source=next(p['path'] for p in json.loads(done.read_text())['outputs'] if p['path'].endswith('__(Vocals).wav'))
        folder=ROOT/'dereverb'/item['id'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists(): continue
        audio,sr=sf.read(source,dtype='float32',always_2d=True)
        start=max(range(0,max(1,int(len(audio)/sr)-20),5),key=lambda s:np.mean(audio[s*sr:(s+20)*sr]**2))
        untreated=audio[start*sr:(start+20)*sr]
        original=folder/'untreated.wav';sf.write(original,untreated,sr,subtype='FLOAT')
        engine.output_dir=str(folder);engine.model_instance.output_dir=str(folder)
        paths=engine.separate(str(original));stems=read_outputs(paths,folder,sr,len(untreated),task='cleanup')
        if 'dry' not in stems: raise RuntimeError(f'No dry stem: {paths}')
        dry=folder/'dry.wav';sf.write(dry,stems['dry'].T,sr,subtype='FLOAT')
        write_json(folder/'complete.json',dict(source=fingerprint(source),start=start,seconds=20,
            model=FUSED_DEREVERB,untreated=fingerprint(original),dry=fingerprint(dry),
            adoption='No automatic adoption. Check loss of rasp, breath, consonants and tails by listening.'))
        print(item['id'],'dereverb control complete',flush=True)

if __name__=='__main__':main()
