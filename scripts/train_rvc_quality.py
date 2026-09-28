"""Isolated, matched Shinedown adaptation runs and Seed-VC training data preparation."""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.benchmark_rvc_quality import ROOT, OUTPUTS, fingerprint, write_json


def prepare(root, variant):
    import numpy as np
    import soundfile as sf
    corpus=json.loads((root/'corpus.json').read_text())['corpus']
    dataset=root/'training'/variant
    sources=[]
    for item in corpus:
        if item['voice']!='shinedown' or item['split']=='test': continue
        if variant=='refreshed':
            done=json.loads((root/'separation'/item['id']/'complete.json').read_text())
            source=Path(next(p['path'] for p in done['outputs'] if p['path'].endswith('__(Vocals).wav')))
        else:
            token={'shine_devil':'sd_devil_clean','shine_cut_the_cord':'Cut The Cord',
                   'shine_simple_man':'Simple Man','shine_state_of_my_head':'State Of My Head',
                   'shine_monsters':'MONSTERS','shine_second_chance':'Second Chance',
                   'shine_sound_of_madness':'Sound Of Madness'}[item['id']]
            source=next(p for p in (OUTPUTS/'voices/Shinedown666/raw').glob('*.wav') if token in p.name)
        dest=dataset/item['split']/source.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        sources.append(dict(id=item['id'],split=item['split'],source=fingerprint(source),copy=fingerprint(dest)))
    write_json(dataset/'sources.json',sources)
    return dataset


def features(directory):
    from modules.rvc.infer.modules.train.preprocess import preprocess_trainset
    from modules.rvc.infer.modules.train.extract.extract_f0_print import extract_f0_features
    from modules.rvc.infer.modules.train.extract_feature_print import extract_feature_print
    for split in ('train','validation'):
        exp=directory/split/'features'
        if not (exp/'filelist.txt').exists():
            # Inputs are collected separately: avoid recursively processing feature outputs.
            inp=directory/(split+'_inputs')
            inp.mkdir(exist_ok=True)
            for p in (directory/split).glob('*.wav'): shutil.copy2(p,inp/p.name)
            preprocess_trainset(str(inp),48000,1,str(exp),3.0)
            extract_f0_features(str(exp),1,'rmvpe')
            extract_feature_print('cuda',str(exp),'v2',False)
            rows=[]
            for wav in sorted((exp/'0_gt_wavs').glob('*.wav')):
                feat=exp/'3_feature768'/f'{wav.stem}.npy'
                pitch=exp/'2a_f0'/f'{wav.name}.npy'
                pitchf=exp/'2b-f0nsf'/f'{wav.name}.npy'
                if not all(p.exists() for p in (feat,pitch,pitchf)):
                    raise RuntimeError(f'Missing features for {wav}')
                rows.append('|'.join(map(str,(wav,feat,pitch,pitchf)))+'|0')
            if not rows: raise RuntimeError('Empty training set')
            (exp/'filelist.txt').write_text('\n'.join(rows),encoding='utf-8')


def train(root, variant, epochs):
    import torch
    import random
    import numpy as np
    random.seed(20260924); np.random.seed(20260924); torch.manual_seed(20260924)
    torch.set_num_threads(4)
    directory=prepare(root,variant)
    features(directory)
    from modules.rvc.infer.lib.train.utils import HParams
    from modules.rvc.infer.modules.train import train as trainer
    from modules.rvc.infer.lib.train import process_ckpt
    # Both arms start from the exact same trained G/D weights with fresh optimizers.
    # Validation is held out from adaptation, not from the historical epoch-75 run.
    config=json.loads((OUTPUTS/'voices/Shinedown666/config.json').read_text())
    config['train'].update(epochs=epochs,batch_size=4,disable_auto_stop=True,seed=20260924,
                           learning_rate=2e-5,lr_decay=.999,log_interval=100)
    exp=directory/'run'
    (exp/'saves').mkdir(parents=True,exist_ok=True)
    config.update(model_dir=str(exp),experiment_dir=str(exp),name=f'Shinedown666_{variant}_adapt',
                  version='v2',sample_rate='48k',if_f0=1,gpus='0',
                  pretrainG=str(OUTPUTS/'voices/Shinedown666/saves/G_e75.pth'),
                  pretrainD=str(OUTPUTS/'voices/Shinedown666/saves/D_e75.pth'),
                  save_epoch_frequency=5,save_latest_only=False,save_every_weights=1,
                  save_every_steps=0,if_cache_data_in_gpu=0,total_epoch=epochs)
    config['data']['training_files']=str(directory/'train/features/filelist.txt')
    config['data']['validation_files']=str(directory/'validation/features/filelist.txt')
    write_json(exp/'config.json',config)
    # Limit all trainer exports to this study, including periodic exports.
    trainer.model_path=process_ckpt.model_path=str(root/'model_store')
    (root/'model_store/trained').mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    trainer.run(0,1,HParams(**config),logging.getLogger('rvc-quality'),lambda *args: None)
    write_json(exp/'complete.json',dict(epochs=epochs,variant=variant,
                initialization=fingerprint(config['pretrainG']),
                limitation='Validation song was present in historical baseline training; held out from adaptation only.'))


def seed_data(root):
    import soundfile as sf
    import numpy as np
    voices={'huxlxy':list((OUTPUTS/'voices/huxlxy_v2/raw').glob('*.wav'))}
    sources=root/'training/refreshed/sources.json'
    if sources.exists():
        voices['shinedown']=[Path(p['copy']['path']) for p in json.loads(sources.read_text()) if p['split']=='train']
    for voice,paths in voices.items():
        dest=root/'seed_training'/voice/'data'
        dest.mkdir(parents=True,exist_ok=True)
        manifest=[]
        for source in paths:
            # Huxlxy: reserve a complete original song from adaptation.
            if voice=='huxlxy' and source.stem=='acid_rain_vox': continue
            audio,sr=sf.read(source,dtype='float32',always_2d=True)
            source_identity=fingerprint(source)
            for i,start in enumerate(range(0,len(audio),10*sr)):
                clip=audio[start:start+10*sr].mean(axis=1)
                if len(clip)<sr or np.sqrt(np.mean(clip**2))<.003: continue
                target=dest/f'{source.stem}_{i:03d}.wav'
                same=False
                if target.exists():
                    previous,previous_sr=sf.read(target,dtype='float32')
                    same=previous_sr==sr and np.array_equal(previous,clip)
                if not same:sf.write(target,clip,sr,subtype='FLOAT')
                manifest.append(dict(source=str(source),source_identity=source_identity,start=start/sr,clip=fingerprint(target)))
        write_json(dest.parent/'manifest.json',manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['original','refreshed','seed-data'])
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--epochs',type=int,default=25)
    a=p.parse_args()
    if a.action=='seed-data': seed_data(a.root)
    else: train(a.root,a.action,a.epochs)
