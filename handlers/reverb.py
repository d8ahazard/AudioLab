"""Public file-based entry points for validated stereo reverb capture/restoration."""
from pathlib import Path

def extract_reverb(dry_path, wet_path, param_output_path, **settings):
    """Capture a wet-only, causal stereo response with held-out validation.

    wet_path is the removed effect (original minus estimated dry), not the full mix.
    Legacy tuning arguments are intentionally unsupported: recapture using the new fit.
    """
    from modules.reverb_ir import capture_reverb
    return capture_reverb(dry_path, wet_path, param_output_path, **settings)


def apply_reverb(dry_path, param_path, output_path, **settings):
    """Restore a validated stereo effect without normalization or clipping."""
    from modules.reverb_ir import restore_reverb
    return restore_reverb(dry_path, param_path, output_path, **settings)


def process_song(dry_path, wet_path, output_dir):
    """Capture from dry + wet-only effect files, then restore the validated effect."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    params = folder / "reverb_params.json"
    output = folder / "reverb_applied.wav"
    extract_reverb(dry_path, wet_path, params)
    return apply_reverb(dry_path, params, output)
