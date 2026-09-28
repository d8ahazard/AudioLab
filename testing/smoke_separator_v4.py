"""Real inference coverage on generated excerpts, isolated from user projects."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
from modules.separator.stem_separator import separate_music
from modules.separator.model_runtime import AudioLabSeparator
from modules.separator.stem_manifest import write_json


def main():
    root = ROOT / "outputs/separation_v4_validation"
    folder = root / "operational"
    folder.mkdir(exist_ok=True)
    source = next((root / "excerpts").glob("*.wav"))
    audio, sr = sf.read(source, dtype="float32", always_2d=True)
    short = folder / "short.wav"
    sf.write(short, audio[:sr * 8], sr, subtype="FLOAT")
    jobs = {
        "hybrid_frazer_default": dict(separate_bg_vocals=True),
        "fast_bg_layers": dict(separation_profile="v4", separation_quality="fast", separate_bg_vocals=True, bg_vocal_layers=2),
        "maximum": dict(separation_profile="v4", separation_quality="maximum", separate_bg_vocals=False),
        "legacy_v3": dict(separation_profile="v3", separate_bg_vocals=False),
        "instruments": dict(separation_profile="v4", separation_quality="fast", separate_bg_vocals=False,
                            vocals_only=False, separate_drums=True, separate_woodwinds=True, alt_bass_model=True),
    }
    for name, options in jobs.items():
        out = folder / name
        if (out / "complete.json").exists():
            continue
        out.mkdir(exist_ok=True)
        events = []
        paths = separate_music({str(out): [str(short)]}, callback=lambda progress, *args: events.append(progress), **options)
        assert events and events[-1] == 1 and all(0 <= p <= 1 for p in events)
        for path in paths:
            arr, rate = sf.read(path, always_2d=True)
            assert rate == sr and arr.shape == (sr * 8, 2) and np.isfinite(arr).all()
        write_json(out / "complete.json", {"outputs": paths, "settings": options, "progress": events})
    cpu = folder / "cpu"
    if not (cpu / "complete.json").exists():
        cpu.mkdir(exist_ok=True)
        # Mono, non-44.1kHz, short input exercises upstream resampling/padding.
        mono = cpu / "mono.wav"
        sf.write(mono, np.zeros(1600, dtype=np.float32), 16000, subtype="FLOAT")
        sep = AudioLabSeparator(cpu=True, model_file_dir=str(ROOT / "models/audio_separator"),
                                output_dir=str(cpu), use_soundfile=True, log_level=40)
        sep.load_model("Kim_Vocal_2.onnx")
        assert sep.torch_device.type == "cpu"
        paths = sep.separate(str(mono))
        for path in paths:
            data, rate = sf.read(cpu / path, always_2d=True)
            assert rate == 44100 and data.shape == (4410, 2) and np.isfinite(data).all()
        write_json(cpu / "complete.json", {"outputs": paths, "runs": sep.run_records})


if __name__ == "__main__":
    main()
