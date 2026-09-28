"""Create immutable source inventories and candidate profiles without approving data."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modules.rvc_v3.performer.contracts import identity,write_json
ROOT=Path('D:/AI-outputs/AudioLab/performer_v3_validation')
OLD=Path('D:/AI-outputs/AudioLab/rvc_quality_validation')


def main():
    import soundfile as sf
    sources=Path('D:/AI-outputs/AudioLab/voices');models=Path('D:/models/AudioLab/trained')
    specs=[('tupac','Tupac1K',models/'Tupac1K.pth',models/'Tupac1K.index','rap'),
           ('huxlxy','huxlxy_v2',models/'huxlxy_v2_v500.pth',models/'huxlxy_v2_v500.index','singing'),
           ('shinedown','Shinedown666',OLD/'models/Shinedown666_e75.pth',OLD/'models/Shinedown666_e75.index','singing'),
           ('chester','ChesterRedux',models/'ChesterRedux.pth',models/'ChesterRedux.index','singing')]
    refs=json.loads((OLD/'references.json').read_text());inventory=[]
    for voice,folder,checkpoint,index,mode in specs:
        records=[]
        paths=sorted((sources/folder/'raw').glob('*.wav'))
        for i,path in enumerate(paths):
            info=sf.info(path)
            records.append(dict(recording_id=path.stem,source=identity(path),duration=info.duration,
                split='validation' if i==len(paths)-2 else 'test' if i==len(paths)-1 else 'train',
                target_only_reviewed=voice=='huxlxy',segments=[],
                status='needs_segment_review' if voice!='huxlxy' else 'original_single_performer_stem',
                notes='Exclude guest verses, hooks and backing voices before adaptation.'))
        if voice in refs:reference=refs[voice]['audio']['path']
        elif voice=='chester':reference=str(OLD/'seed_training/chester/reference.wav')
        else:
            raw=next(p for p in paths if 'Dear Mama' in p.name)
            audio,sr=sf.read(raw,dtype='float32',always_2d=True)
            ref=ROOT/'references/tupac.wav';ref.parent.mkdir(parents=True,exist_ok=True)
            if not ref.exists():sf.write(ref,audio[40*sr:52*sr].mean(1),sr,subtype='FLOAT')
            reference=str(ref)
        # Profiles are experimental until recording-level segmentation and reference review pass.
        profile=dict(schema=1,id=voice,mode=mode,checkpoint=identity(checkpoint),index=identity(index),
            reference=identity(reference),reference_reviewed=voice=='huxlxy',reference_text=None,
            recordings=records,stage='m0_candidates',adapters={},
            baseline_settings=dict(index_rate=.5,protect=.5 if voice=='shinedown' else .2,pitch_method='rmvpe+'),
            training_gate='pending_target_segmentation',language='en')
        dest=ROOT/'profiles'/f'{voice}.json'
        if not dest.exists():write_json(dest,profile)
        inventory.extend(records)
    write_json(ROOT/'inventory.json',inventory)
    mix=Path('D:/AI-outputs/AudioLab/downloaded/Lose Yourself.mp3')
    if not mix.is_file():raise FileNotFoundError('Download official Lose Yourself source before preparing rap benchmark')
    from scripts.benchmark_rvc_quality import SETTINGS
    write_json(ROOT/'corpus.json',dict(schema=1,separation=SETTINGS,corpus=[dict(id='tupac_test',voice='tupac',split='test',source=identity(mix),old_lead=None)]))
    write_json(ROOT/'milestones.json',dict(M0='in progress: candidate inventories; review pending',M1='not evaluated',M2='gated on M1',M3='gated on M2',M4='gated on M3',M5='gated on rap and singing evidence'))
    print(ROOT)


if __name__=='__main__':main()
