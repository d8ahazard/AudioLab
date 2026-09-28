"""Local, explicitly unreviewed transcripts for the Vevo2 lyric experiment."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_rvc_quality import ROOT, write_json, fingerprint

def main():
    from faster_whisper import WhisperModel
    model=WhisperModel('distil-large-v3',device='cuda',compute_type='float16',
                      download_root='D:/models/hf-cache/hub',local_files_only=True)
    paths={c['id']:c['source']['path'] for c in json.loads((ROOT/'excerpts.json').read_text())}
    paths.update({v+'_reference':x['audio']['path'] for v,x in json.loads((ROOT/'references.json').read_text()).items()})
    transcripts={}
    for name,path in paths.items():
        dest=ROOT/'transcripts'/f'{name}.json'
        if dest.exists() and json.loads(dest.read_text()).get('source')==fingerprint(path):
            transcripts[name]=json.loads(dest.read_text());continue
        segments,info=model.transcribe(path,language='en',word_timestamps=True,beam_size=5,
                                       condition_on_previous_text=False,vad_filter=False)
        result=[]
        for seg in segments:
            result.append(dict(start=seg.start,end=seg.end,text=seg.text,
                               words=[dict(start=w.start,end=w.end,word=w.word,probability=w.probability) for w in seg.words or []]))
        item=dict(source=fingerprint(path),segments=result,text=' '.join(x['text'].strip() for x in result),
                  reviewed=False,model='Systran/faster-distil-whisper-large-v3',
                  limitation='Automatic singing transcript; errors confound lyric conditioning and must be reviewed.')
        write_json(dest,item);transcripts[name]=item
        print('Transcribed',name,flush=True)
    write_json(ROOT/'transcripts.json',transcripts)

if __name__=='__main__': main()
