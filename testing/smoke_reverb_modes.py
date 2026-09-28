"""Real Sucial Fused dry/capture exports and wet-clone restoration routing."""
import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from modules.separator.stem_separator import EnsembleDemucsMDXMusicSeparationModel as Pipeline
from modules.reverb_ir import capture_for_stem,restore_reverb
from modules.separator.stem_manifest import write_json
from wrappers.separate import Separate

root=REPO/'outputs/separation_v4_validation'
source=root/'background-comparison/predictions/Lorde - Royals US Version_5365a577/karaoke/vocals.wav'
wet,sr=sf.read(source,dtype='float32',always_2d=True)
for mode in ('Dry vocals','Capture reverb'):
    folder=root/'operational/reverb_modes'/mode.replace(' ','_')
    folder.mkdir(parents=True,exist_ok=True)
    model=Pipeline({'vocal_reverb':mode,'smart_stems':'off'})
    result=model._apply_transform_chain(wet.T,sr,'Royals','vocals',str(folder))
    paths=model._save_all_stems({'Royals':{'base_name':'Royals','mix_np':wet.T,'sr':sr,
        'vocals':result,'output_folder':str(folder)}})
    assert Separate._manifest_valid(folder)
    if mode=='Capture reverb':
        np.testing.assert_array_equal(result,wet.T)
        assert capture_for_stem(paths[0],folder) is None
        # A wet clone stand-in verifies merge routing and level behavior, not voice conversion.
        clone=folder/'Royals__(Vocals)(Cloned).wav'
        sf.write(clone,wet,sr,subtype='FLOAT')
        ir=capture_for_stem(clone,folder)
        assert ir
        output=restore_reverb(clone,ir,folder/'Royals__(Vocals)(Cloned)(Re-Reverb).wav')
        restored,rate=sf.read(output,always_2d=True)
        assert restored.shape==wet.shape and rate==sr and np.isfinite(restored).all()
        assert capture_for_stem(output,folder) is None
    else:
        assert not np.allclose(result,wet.T)
        assert capture_for_stem(paths[0],folder) is None
    write_json(folder/'complete.json',{'mode':mode,'outputs':paths,'runs':model.separator.run_records})
    print(mode+' passed',flush=True)
