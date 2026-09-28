"""Whole-song transcription context for benchmark review, without changing frozen lyrics."""
import json
from pathlib import Path
import sys
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modules.rvc_v3.performer.contracts import identity,write_json
ROOT=Path('D:/AI-outputs/AudioLab/performer_v3_validation')
OLD=Path('D:/AI-outputs/AudioLab/rvc_quality_validation')


def run(device):
    from faster_whisper import WhisperModel
    # CPU audit can run while the single GPU queue is occupied; bound threads and memory.
    model=WhisperModel('distil-large-v3',device=device,compute_type='int8' if device=='cpu' else 'float16',cpu_threads=4,
        download_root='D:/models/hf-cache/hub',local_files_only=True)
    for voice,source_root,name in [('tupac',ROOT,'tupac_test'),('huxlxy',OLD,'hux_test'),('shinedown',OLD,'shine_test'),('chester',OLD,'chester_test')]:
        complete=json.loads((source_root/'separation'/name/'complete.json').read_text())
        source=next(x['path'] for x in complete['outputs'] if x['path'].endswith('__(Vocals).wav'))
        target=ROOT/'source-analysis'/f'{voice}.json'
        if target.exists():continue
        segments,_=model.transcribe(source,language='en',word_timestamps=True,beam_size=5,condition_on_previous_text=False)
        records=[dict(start=w.start,end=w.end,text=w.word.strip(),confidence=w.probability)
                 for segment in segments for w in segment.words or [] if w.end>w.start]
        write_json(target,dict(source=identity(source),words=records,text=' '.join(w['text'] for w in records),
            reviewed=False,purpose='Context for manual correction; not an automatic replacement for frozen benchmark text'))
        print(voice,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if args.device=='cuda':
        from filelock import FileLock
        with FileLock('D:/AI-outputs/AudioLab/rvc_quality_validation/.conversion.lock'):run(args.device)
    else:run(args.device)
