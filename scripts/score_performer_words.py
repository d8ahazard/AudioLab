"""Calibrate automatic word errors against the same RVC baseline, not zero-error assumptions."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modules.rvc_v3.performer.contracts import words,identity,write_json
ROOT=Path('D:/AI-outputs/AudioLab/performer_v3_validation')


def errors(expected,observed):
    a,b=words(expected),words(observed);row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        next_row=[i]
        for j,y in enumerate(b,1):next_row.append(min(next_row[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=next_row
    return dict(edits=row[-1],reference_words=len(a),wer=row[-1]/max(1,len(a)))


def main():
    from filelock import FileLock
    from faster_whisper import WhisperModel
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--round',required=True);args=parser.parse_args()
    root=ROOT/'rounds'/args.round;cases=json.loads((ROOT/'benchmark.json').read_text())
    started=time.monotonic()
    while not all((root/c['id']/'performer/result.json').exists() for c in cases):
        if time.monotonic()-started>600:raise TimeoutError('Round has missing results; word audit did not run')
        time.sleep(3)
    reports=[]
    with FileLock('D:/AI-outputs/AudioLab/rvc_quality_validation/.conversion.lock'):
        model=WhisperModel('distil-large-v3',device='cuda',compute_type='float16',cpu_threads=4,
            download_root='D:/models/hf-cache/hub',local_files_only=True)
        for case in cases:
            folder=root/case['id'];p=json.loads((folder/'performer/result.json').read_text());b=json.loads((folder/'preserve/result.json').read_text())
            expected=' '.join(x['text'] for x in p['phrases'])
            segments,_=model.transcribe(b['output'],language='en',beam_size=5,condition_on_previous_text=False,word_timestamps=True)
            text=' '.join(s.text.strip() for s in segments)
            report=dict(case=case['id'],expected=expected,expected_reviewed=False,
                baseline=dict(audio=b['output_identity'],text=text,**errors(expected,text)),
                performer=dict(audio=p['output_identity'],text=p['output_word_check']['observed'],**errors(expected,p['output_word_check']['observed'])),
                manually_reviewed=False)
            write_json(folder/'word-metrics.json',report);reports.append(report)
    write_json(root/'word-metrics.json',reports)
    print([(r['case'],r['baseline']['wer'],r['performer']['wer']) for r in reports])


if __name__=='__main__':main()
