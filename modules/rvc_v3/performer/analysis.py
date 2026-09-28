"""Observed word timing and voicing, with explicit automatic-label provenance."""
from pathlib import Path
from .contracts import identity,write_json,words,validate_edits,FitError


def transcribe(path):
    from faster_whisper import WhisperModel
    model=WhisperModel('distil-large-v3',device='cuda',compute_type='float16',download_root='D:/models/hf-cache/hub',local_files_only=True)
    segments,_=model.transcribe(str(path),language='en',word_timestamps=True,beam_size=5,condition_on_previous_text=False,vad_filter=False)
    records=[]
    for segment in segments:
        records.extend(dict(start=w.start,end=w.end,text=w.word.strip(),confidence=w.probability) for w in segment.words or [] if w.end>w.start)
    del model
    return dict(source=identity(path),text=' '.join(w['text'] for w in records),words=records,reviewed=False,alignment='automatic_whisper_words')


def align_corrected(path,text,duration,device='cuda'):
    """Realign corrected preview text; never borrow timestamps from different words."""
    if duration>20:raise FitError('Corrected full-song text requires reviewed phrase alignment; preview a window up to 20 seconds')
    import av.video
    from whisperx.alignment import load_align_model,align
    model,metadata=load_align_model('en',device,model_dir='D:/models/AudioLab-performer/alignment')
    result=align([dict(start=0.,end=duration,text=text)],model,metadata,str(path),device,interpolate_method='nearest')
    records=[]
    for word in result['word_segments']:
        import math
        if not all(k in word and math.isfinite(word[k]) for k in ('start','end','score')) or not 0<=word['start']<word['end']<=duration+.04 or word['score']<.1:
            raise FitError(f"Cannot reliably align {word.get('word')!r}: start={word.get('start')}, end={word.get('end')}, score={word.get('score')}; review this word and its phrase window")
        # CTC's final frame can extend one stride beyond the input samples.
        end=min(duration,word['end'])
        if end<=word['start']:raise FitError('Alignment placed a word beyond the source')
        records.append(dict(start=word['start'],end=end,text=word['word'],confidence=word['score'],raw_end=word['end']))
    if words(' '.join(w['text'] for w in records))!=words(text):raise FitError('Alignment omitted corrected words')
    del model
    return dict(source=identity(path),text=text,words=records,reviewed=False,text_origin='user_corrected',alignment='forced_wav2vec2_words')


def phrase_windows(analysis,duration,request):
    if request.source_transcript and words(request.source_transcript)!=words(analysis['text']):
        raise ValueError('Corrected transcript differs from automatic alignment; realign before generating, rather than assigning incorrect word times')
    edits=validate_edits(request.phrase_edits,duration)
    if edits:
        # Explicit source gaps remain target-voice V2 in the worker; edited windows are generated.
        return [dict(id=f'phrase_{i}',start=e.start,end=e.end,text=e.text,edited=True) for i,e in enumerate(edits)]
    observed=analysis['words']
    if not observed:raise FitError('No reliably timed words in source')
    if request.target_lyrics:
        if duration>20:raise FitError('Whole-song lyric replacements require phrase edits; use bounded windows')
        return [dict(id='phrase_0',start=0.,end=duration,text=request.target_lyrics,edited=words(request.target_lyrics)!=words(analysis['text']))]
    groups=[];current=[]
    for word in observed:
        if current and (word['start']-current[-1]['end']>.45 or word['end']-current[0]['start']>14):
            groups.append(current);current=[]
        current.append(word)
    if current:groups.append(current)
    result=[]
    for i,group in enumerate(groups):
        start=max(0.,group[0]['start']-.04);end=min(duration,group[-1]['end']+.04)
        if result and start<result[-1]['end']:
            boundary=(start+result[-1]['end'])/2;result[-1]['end']=boundary;start=boundary
        result.append(dict(id=f'phrase_{i}',start=start,end=end,text=' '.join(w['text'] for w in group),edited=False))
    return result
