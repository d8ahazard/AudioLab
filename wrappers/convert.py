import os
from typing import List, Dict, Any, Callable

from ffmpeg_progress_yield import FfmpegProgress

from handlers.config import output_path
from wrappers.base_wrapper import BaseWrapper, TypedInput


class Convert(BaseWrapper):
    priority = 10
    title = "Convert"
    default = True
    allowed_kwargs = {
        "bitrate": TypedInput(
            description="Bitrate for the output MP3 file",
            default="192k",  # Default bitrate used by FFMPEG when unspecified
            type=str,
            gradio_type="Dropdown",
            choices=["64k", "96k", "128k", "160k", "192k", "224k", "256k", "320k"],
        ),
    }

    def register_api_endpoint(self, api) -> Any:
        pass

    def process_audio(self, inputs: List[str], callback: Callable = None, **kwargs: Dict[str, Any]) -> List[str]:
        output_folder = os.path.join(output_path, "converted")
        os.makedirs(output_folder, exist_ok=True)
        bitrate = kwargs.get("bitrate", "192k")  # Default bitrate

        # Filter inputs and initialize progress tracking
        inputs, outputs = self.filter_inputs(inputs, "audio")
        # Remove mp3s
        inputs = [i for i in inputs if not i.endswith(".mp3")]
        total_files = len(inputs)
        total_steps = total_files  # Each file counts as one long-running step
        current_step = 0

        if callback:
            callback(0, f"Starting conversion of {total_files} files", total_steps)

        for i, file in enumerate(inputs):
            file_name, ext = os.path.splitext(os.path.basename(file))
            if ext.lower() in [".wav", ".flac", ".ogg", ".m4a"]:  # Ensure supported formats
                out_file = os.path.join(output_folder, f"{file_name}.mp3")
                cmd = ["ffmpeg", "-i", file, "-b:a", bitrate, out_file]

                try:
                    progress_tracker = FfmpegProgress(cmd)
                    for progress in progress_tracker.run_command_with_progress():
                        if callback:
                            callback(current_step + progress / 100, f"Converting {file_name}", total_steps)

                    print(f"Processed file: {out_file}")
                    inputs[i] = out_file
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
                    continue

            # Increment step count after each file
            current_step += 1
            if callback:
                callback(current_step, f"Completed {file_name}", total_steps)

        inputs += outputs

        if callback:
            callback(total_steps, "Conversion complete", total_steps)

        return inputs
