import os
import subprocess
from typing import List, Dict, Any, Optional

from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput


class Convert(BaseWrapper):
    priority = 10
    title = "Convert"
    default = True
    description = "Convert audio files to MP3 format."
    allowed_kwargs = {
        "bitrate": TypedInput(
            description="Bitrate for the output MP3 file",
            default="320k",  # Default bitrate used by FFMPEG when unspecified
            type=str,
            gradio_type="Dropdown",
            choices=["64k", "96k", "128k", "160k", "192k", "224k", "256k", "320k"],
        ),
    }

    def register_api_endpoint(self, api) -> Any:
        """
        Register FastAPI endpoint for audio format conversion.
        
        Args:
            api: FastAPI application instance
            
        Returns:
            The registered endpoint route
        """
        from fastapi import Body
        
        # Create models for JSON API
        FileData, JsonRequest = self.create_json_models()

        @api.post("/api/v1/process/convert", tags=["Audio Processing"])
        async def process_convert_json(
            request: JsonRequest = Body(...)
        ):
            """
            Convert audio files to MP3 format.
            
            This endpoint converts audio files to MP3 format with configurable bitrate settings.
            It provides a simple way to standardize your audio collection to a consistent format
            while maintaining quality control through bitrate selection.
            
            ## Request Body
            
            ```json
            {
              "files": [
                {
                  "filename": "audio.wav",
                  "content": "base64_encoded_file_content..."
                }
              ],
              "settings": {
                "bitrate": "192k"
              }
            }
            ```
            
            ## Parameters
            
            - **files**: Array of file objects, each containing:
              - **filename**: Name of the file (with extension)
              - **content**: Base64-encoded file content
            - **settings**: Conversion settings with the following options:
              - **bitrate**: Bitrate for the output MP3 file (default: "320k")
                - Options: "64k", "96k", "128k", "160k", "192k", "224k", "256k", "320k"
                - Higher bitrates provide better audio quality but larger file sizes
            
            ## Response
            
            ```json
            {
              "files": [
                {
                  "filename": "audio.mp3",
                  "content": "base64_encoded_file_content..."
                }
              ]
            }
            ```
            
            The API returns an object containing an array of files, each with filename and Base64-encoded content.
            """
            # Use the handle_json_request helper from BaseWrapper
            return self.handle_json_request(request, self.process_audio)

        return process_convert_json

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        bitrate = kwargs.get("bitrate", "192k")  # Default bitrate

        # Filter inputs and initialize progress tracking
        pj_outputs = []
        for project in inputs:
            outputs = []
            input_files, _ = self.filter_inputs(project, "audio")
            non_mp3_inputs = [i for i in input_files if not i.endswith(".mp3")]
            if not non_mp3_inputs:
                continue
            output_folder = os.path.join(project.project_dir)
            os.makedirs(output_folder, exist_ok=True)
            for idx, input_file in enumerate(non_mp3_inputs):
                if callback is not None:
                    pct_done = int((idx + 1) / len(non_mp3_inputs))
                    callback(pct_done, f"Converting {os.path.basename(input_file)}", len(non_mp3_inputs))
                file_name, ext = os.path.splitext(os.path.basename(input_file))
                output_file = os.path.join(output_folder, f"{file_name}.mp3")
                if os.path.exists(output_file):
                    os.remove(output_file)
                # Convert to MP3
                subprocess.run(
                    f'ffmpeg -i "{input_file}" -b:a {bitrate} "{output_file}"',
                    shell=True,
                    stdout=subprocess.DEVNULL,  # Suppress stdout
                    stderr=subprocess.PIPE,  # Redirect stderr to capture errors (optional)
                )
                outputs.append(output_file)
            project.add_output("converted", outputs)
            pj_outputs.append(project)
        return pj_outputs
        
    def test(self):
        """
        Test method for the Convert wrapper.
        
        This method creates a test WAV file and converts it to MP3 format to verify
        the conversion functionality works correctly.
        """
        print("Running Convert wrapper test...")
        
        # Create temporary directory for test
        import tempfile
        import shutil
        import numpy as np
        from scipy.io import wavfile
        from util.data_classes import ProjectFiles
        import os.path
        
        temp_dir = tempfile.mkdtemp(prefix="audiolab_convert_test_")
        print(f"Created temporary directory at: {temp_dir}")
        
        try:
            # Create a simple test signal
            duration = 2  # seconds
            sample_rate = 44100
            t = np.linspace(0, duration, sample_rate * duration)
            
            # Generate a simple sine wave
            signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz A note
            
            # Save as WAV file
            wav_path = os.path.join(temp_dir, "test_audio.wav")
            wavfile.write(wav_path, sample_rate, signal.astype(np.float32))
            
            print(f"Created test WAV file at: {wav_path}")
            
            # Create ProjectFiles object for testing
            project = ProjectFiles(wav_path)
            project.project_dir = temp_dir
            
            # Define a simple callback function
            def test_callback(progress, message, total):
                print(f"Progress: {progress}/{total} - {message}")
            
            # Process the test data with different bitrates
            for bitrate in ["128k", "320k"]:
                print(f"\nTesting conversion with bitrate: {bitrate}")
                results = self.process_audio([project], callback=test_callback, bitrate=bitrate)
                
                if not results:
                    raise Exception("Test failed: No results returned")
                
                # Check if output file was created
                project_output = results[0]
                if len(project_output.last_outputs) == 0:
                    raise Exception("Test failed: No output files created")
                    
                mp3_output = project_output.last_outputs[0]
                if not os.path.exists(mp3_output):
                    raise Exception(f"Test failed: Output MP3 file not found at {mp3_output}")
                    
                # Check if output file has .mp3 extension
                if not mp3_output.endswith(".mp3"):
                    raise Exception(f"Test failed: Output file does not have .mp3 extension: {mp3_output}")
                
                # Check file size to ensure it's not empty
                file_size = os.path.getsize(mp3_output)
                if file_size == 0:
                    raise Exception(f"Test failed: Output MP3 file is empty: {mp3_output}")
                    
                print(f"Test successful! MP3 file created at: {mp3_output} (size: {file_size} bytes)")
                
                # Clean up MP3 file before next test
                os.remove(mp3_output)
                
        except Exception as e:
            print(f"Test error: {e}")
            raise
        finally:
            # Clean up the temporary directory
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up temporary directory: {cleanup_error}")
        
        print("Convert wrapper test completed successfully!")
