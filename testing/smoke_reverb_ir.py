"""Exercise real model cleanup -> capture -> manifest -> restoration gate."""
import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from modules.separator.stem_separator import EnsembleDemucsMDXMusicSeparationModel as Pipeline
from modules.reverb_ir import capture_for_stem, restore_reverb, InvalidImpulseResponse
from modules.separator.stem_manifest import write_json

root = REPO/'outputs/separation_v4_validation'
folder = root/'operational/stereo_ir'
folder.mkdir(parents=True,exist_ok=True)
source = root/'background-comparison/predictions/Lorde - Royals US Version_5365a577/karaoke/vocals.wav'
audio, sr = sf.read(source,dtype='float32',always_2d=True)
model = Pipeline({'separation_profile':'hybrid_cleaned','store_reverb_ir':True,
                  'reverb_removal':'Main Vocals','smart_stems':'off'})
dry = model._apply_transform_chain(audio.T,sr,'Royals','vocals',str(folder))
paths = model._save_all_stems({'Royals':{'base_name':'Royals','mix_np':audio.T,'sr':sr,
    'vocals':dry,'output_folder':str(folder)}})
assert len(paths)==1
capture_path = capture_for_stem(folder/'Royals__(Vocals)(Cloned).wav',folder)
assert capture_path
capture = json.loads(Path(capture_path).read_text())
if capture['valid_for_restore']:
    output = restore_reverb(paths[0],capture_path,folder/'reapplied.wav')
    arr,rate=sf.read(output,always_2d=True)
    assert arr.shape==audio.shape and rate==sr and np.isfinite(arr).all()
else:
    try:
        restore_reverb(paths[0],capture_path,folder/'reapplied.wav')
    except InvalidImpulseResponse:
        pass
    else:
        raise AssertionError('Invalid fit was not blocked')
write_json(folder/'complete.json',{'capture':capture_path,'valid_for_restore':capture['valid_for_restore'],
    'diagnostics':capture['diagnostics'],'outputs':paths,'runs':model.separator.run_records})
print('Real cleanup, capture, per-stem manifest and restore gate passed')
