"""Opt-in Mega inference in an isolated process; no app-global loader patches."""
import argparse
import json
import logging
import re
from pathlib import Path
from unittest.mock import patch
from modules.separator.model_runtime import AudioLabSeparator, download_verified
from modules.separator.instrument_policy import MEGA_MODEL, MEGA_CONFIG


class MegaSeparator(AudioLabSeparator):
    def download_model_files(self, name):
        if name != MEGA_MODEL:
            return super().download_model_files(name)
        base = 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.21/'
        for filename, digest in [(MEGA_MODEL, 'c62820893bbf86d4e734f966bd142d9157cfc8bb8e79e9d8f9ea553f3ff3519f'),
                                 (MEGA_CONFIG, '7e198062a251587088adb91215a4f44ab59e67bd62fcc805cf54d6e7dfc51103')]:
            download_verified(base+filename, Path(self.model_file_dir)/filename, digest)
        self.model_is_uvr_vip = False
        self.model_friendly_name = name
        return name, 'MDXC', name, str(Path(self.model_file_dir)/name), MEGA_CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source'); parser.add_argument('output'); parser.add_argument('models')
    parser.add_argument('--quality', default='balanced'); parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    sep = MegaSeparator(model_file_dir=args.models, output_dir=args.output, quality=args.quality,
        cpu=args.cpu, use_autocast=not args.cpu, use_soundfile=True, preserve_gain=True, log_level=logging.ERROR)
    from audio_separator.separator.roformer.parameter_validator import ParameterValidator
    with patch.dict(ParameterValidator.PARAMETER_RANGES, {'num_stems': (53, 53)}):
        sep.load_model(MEGA_MODEL)
    outputs = {}
    for name in sep.separate(args.source):
        p = Path(name)
        if not p.is_absolute(): p = Path(args.output)/p
        role = re.findall(r'\(([^()]*)\)', p.stem)[0].lower()
        outputs[role] = p.name
    if len(outputs) != 53:
        raise RuntimeError('Mega did not produce its 53 configured outputs')
    (Path(args.output)/'mega.json').write_text(json.dumps({'outputs':outputs,'runs':sep.run_records}),encoding='utf-8')


if __name__ == '__main__': main()
