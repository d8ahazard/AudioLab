"""Reproducible local comparison runner; never writes into source projects.

Prepare once, run baseline with each environment, then run candidates/fusion.
All predictions are keyed by source content, package, settings and model hashes.
"""
import argparse
import importlib.util
import json
import sys
import time
import itertools
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare(root):
    import librosa
    import numpy as np
    import soundfile as sf
    corpus = json.loads((root / "corpus.json").read_text())
    for item in corpus:
        target = root / "excerpts" / (item["name"] + ".wav")
        if target.exists():
            continue
        audio, sr = librosa.load(item["source"], sr=44100, mono=False)
        duration = audio.shape[-1] / sr
        starts = [max(0, duration * .15), max(0, duration * .5), max(0, duration - 10)]
        clips = [audio[..., int(t * sr):int(t * sr) + 8 * sr] for t in starts]
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, np.concatenate(clips, axis=-1).T, sr, subtype="FLOAT")
        item["excerpt_starts_seconds"] = starts
    (root / "corpus.json").write_text(json.dumps(corpus, indent=2))


def baseline(root, profile):
    from importlib.metadata import version
    import torch
    for name, file in [("modules.separator.separation_profiles", "separation_profiles.py"),
                       ("handlers.patch_separate", "patch_separate.py"), ("baseline_separator", "stem_separator.py")]:
        spec = importlib.util.spec_from_file_location(name, root / "baseline_code" / file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    output = root / ("baseline-" + version("audio-separator") + "-" + profile)
    for source in sorted((root / "excerpts").glob("*.wav")):
        folder = output / source.stem
        if (folder / "complete.json").exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        try:
            paths = module.separate_music({str(folder): [str(source)]}, separation_profile=profile,
                                          separate_bg_vocals=False, vocals_only=True)
            data = {"paths": paths, "seconds": time.perf_counter() - start,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated()}
            (folder / "complete.json").write_text(json.dumps(data, indent=2))
        except Exception as exc:
            (folder / "error.txt").write_text(repr(exc))
            raise


CANDIDATES = [
    "bs_roformer_vocals_resurrection_unwa.ckpt", "melband_roformer_big_beta7.ckpt",
    "melband_roformer_big_beta6x.ckpt", "mel_band_roformer_instrumental_becruily.ckpt",
    "bs_roformer_instrumental_resurrection_unwa.ckpt", "mel_band_roformer_instrumental_fv7z_gabox.ckpt",
    "melband_roformer_becruily_deux.ckpt",
]


def candidates(root, quality="balanced"):
    import logging
    import tempfile
    import librosa
    import soundfile as sf
    from importlib.metadata import version
    from modules.separator.model_runtime import AudioLabSeparator, read_outputs
    from modules.separator.audio_quality import stereo
    from modules.separator.stem_manifest import file_hash, model_fingerprint, write_json
    separator = AudioLabSeparator(model_file_dir=str(ROOT / "models/audio_separator"), quality=quality,
        use_soundfile=True, use_autocast=True, log_level=logging.ERROR, normalization_threshold=1, amplification_threshold=0)
    for name in CANDIDATES:
        separator.load_model(name)
        model_identity = model_fingerprint(separator.model_file_dir, [name])
        for source in sorted((root / "excerpts").glob("*.wav")):
            folder = root / "predictions" / quality / source.stem / name
            identity = {"input": file_hash(source), "package": version("audio-separator"), "model": model_identity, "quality": quality}
            complete = folder / "complete.json"
            if complete.exists() and json.loads(complete.read_text())["identity"] == identity:
                continue
            folder.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=folder) as temp:
                separator.output_dir = temp
                separator.model_instance.output_dir = temp
                paths = separator.separate(str(source))
                audio, sr = librosa.load(source, sr=44100, mono=False)
                stems = read_outputs(paths, temp, sr, audio.shape[-1])
                if not {"vocals", "instrumental"} <= stems.keys():
                    raise RuntimeError(f"Missing primary stems for {name}: {list(stems)}")
                for role in ("vocals", "instrumental"):
                    sf.write(folder / f"{role}.wav", stereo(stems[role]).T, sr, subtype="FLOAT")
            write_json(complete, {"identity": identity, "run": separator.run_records[-1]})


def fusion(root, quality="balanced"):
    import numpy as np
    import soundfile as sf
    from modules.separator.audio_quality import fuse
    from modules.separator.stem_manifest import write_json
    algorithms = ["avg_wave", "avg_complex", "median_magnitude", "min_magnitude", "max_magnitude"]
    # Fixed Resurrection anchor + a vocal/dual candidate + an instrumental candidate.
    recipes = [[CANDIDATES[0], vocal, inst] for vocal in [CANDIDATES[1], CANDIDATES[2], CANDIDATES[6]]
               for inst in CANDIDATES[3:6]]
    index = []
    for source in sorted((root / "excerpts").glob("*.wav")):
        predictions = root / "predictions" / quality / source.stem
        pool = {m: {role: sf.read(predictions / m / f"{role}.wav", dtype="float32", always_2d=True)[0].T
                    for role in ("vocals", "instrumental")} for m in CANDIDATES}
        for models, algorithm in itertools.product(recipes, algorithms):
            key = hashlib.sha256(json.dumps([models, algorithm]).encode()).hexdigest()[:10]
            for role in ("vocals", "instrumental"):
                folder = root / "fusion" / quality / source.stem / key
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / f"{role}.wav"
                if not target.exists():
                    sf.write(target, fuse([pool[m][role] for m in models], algorithm).T, 44100, subtype="FLOAT")
                index.append({"song": source.stem, "models": models, "algorithm": algorithm, "role": role, "path": str(target)})
    write_json(root / "fusion-index.json", index)


def specialists(root):
    import logging
    import tempfile
    import soundfile as sf
    from modules.separator.model_runtime import AudioLabSeparator, BG_MODELS, INSTRUMENT_MODELS, read_outputs
    from modules.separator.audio_quality import fuse
    from modules.separator.stem_manifest import write_json
    separator = AudioLabSeparator(model_file_dir=str(ROOT / "models/audio_separator"), use_soundfile=True,
                                  use_autocast=True, log_level=logging.ERROR)
    trio = [CANDIDATES[0], CANDIDATES[1], CANDIDATES[3]]
    for source in sorted((root / "excerpts").glob("*.wav")):
        pool = root / "predictions/balanced" / source.stem
        inputs = {role: fuse([sf.read(pool / m / f"{role}.wav", dtype="float32", always_2d=True)[0].T for m in trio])
                  for role in ("vocals", "instrumental")}
        inputs["mix"] = sf.read(source, dtype="float32", always_2d=True)[0].T
        jobs = [(m, "vocals", "karaoke" if key == "karaoke" else "bve") for key, m in BG_MODELS.items()]
        jobs += [(m, feed, "instruments") for m in INSTRUMENT_MODELS.values() for feed in ("mix", "instrumental")]
        for model, feed, task in jobs:
            folder = root / "specialists" / source.stem / (model + "-" + feed)
            if (folder / "complete.json").exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            separator.load_model(model)
            with tempfile.TemporaryDirectory(dir=folder) as temp:
                inp = Path(temp) / "input.wav"
                sf.write(inp, inputs[feed].T, 44100, subtype="FLOAT")
                separator.output_dir = temp
                separator.model_instance.output_dir = temp
                stems = read_outputs(separator.separate(str(inp)), temp, 44100, inputs[feed].shape[-1], task)
                for role, audio in stems.items():
                    sf.write(folder / f"{role}.wav", audio.T, 44100, subtype="FLOAT")
            write_json(folder / "complete.json", {"model": model, "feed": feed, "run": separator.run_records[-1]})


def full_songs(root):
    """Full-length experimental candidate; listening scores still gate promotion."""
    from modules.separator.stem_separator import separate_music
    from modules.separator.stem_manifest import write_json
    for item in json.loads((root / "corpus.json").read_text()):
        folder = root / "full-candidate" / item["name"]
        if (folder / "complete.json").exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        paths = separate_music({str(folder): [item["source"]]}, separation_profile="v4",
                               separation_quality="balanced", separate_bg_vocals=True, smart_stems="conservative")
        write_json(folder / "complete.json", {"paths": paths, "seconds": time.perf_counter() - start, "promoted": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "baseline", "candidates", "fusion", "specialists", "full"])
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/separation_v4_validation")
    parser.add_argument("--profile", default="v2")
    parser.add_argument("--quality", default="balanced", choices=["fast", "balanced", "maximum"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root)
    elif args.action == "baseline":
        baseline(args.root, args.profile)
    elif args.action == "candidates":
        candidates(args.root, args.quality)
    elif args.action == "fusion":
        fusion(args.root, args.quality)
    elif args.action == "specialists":
        specialists(args.root)
    elif args.action == "full":
        full_songs(args.root)


if __name__ == "__main__":
    main()
