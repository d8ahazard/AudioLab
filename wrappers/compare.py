import hashlib
import os
from typing import Any, List, Dict

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from scipy.signal import resample, stft

from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper


def compute_file_hash(filepath: str, chunk_size: int = 65536) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def generate_output_filename(file1: str, file2: str, output_folder: str) -> str:
    hash1 = compute_file_hash(file1)[:6]  # shorten the hash for readability
    hash2 = compute_file_hash(file2)[:6]
    base1 = os.path.splitext(os.path.basename(file1))[0]
    base2 = os.path.splitext(os.path.basename(file2))[0]

    return os.path.join(
        output_folder,
        f"{base1}_VS_{base2}_{hash1}_{hash2}_comparison.png"
    )


class Compare(BaseWrapper):
    title = "Compare"
    description = "Compare two audio files using time-domain waveforms and spectrogram differences."
    priority = 1000000

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        """
        Compare two unique audio files and produce:
         - Time-domain waveforms
         - Absolute difference waveform
         - Spectrogram-based difference visualization
        """
        # Filter and ensure exactly two unique files
        pj_outputs = []
        for project in inputs:
            outputs = []
            inputs, _ = self.filter_inputs(project, "audio")

            if len(inputs) >= 2:
                msg = f"Expected exactly 2 unique audio files, got {len(inputs)}."
                if callback is not None:
                    callback(0, msg, 1)
                return []

            unique_files = [project.src_file, inputs[0]]

            # Prepare output
            output_folder = os.path.join(project.project_dir, "comparisons")
            os.makedirs(output_folder, exist_ok=True)
            output_file = generate_output_filename(unique_files[0], unique_files[1], output_folder)

            total_steps = 5
            current_step = 0
            if callback is not None:
                callback(0, "Starting audio comparison with spectrogram difference", total_steps)

            sample_rate = 44100
            waveforms = []

            # 1) Load + resample both
            for file in unique_files:
                audio = AudioSegment.from_file(file)
                # Normalization factor for e.g. 16-bit PCM:
                divisor = float(2 ** (8 * audio.sample_width - 1))
                samples = np.array(audio.get_array_of_samples()) / divisor

                # Resample to common sample_rate
                resampled = resample(samples, int(len(samples) * sample_rate / audio.frame_rate))
                waveforms.append(resampled)

                current_step += 1
                if callback is not None:
                    callback(current_step, f"Loaded/resampled {os.path.basename(file)}", total_steps)

            # 2) Align to min length
            min_len = min(len(w) for w in waveforms)
            waveforms = [w[:min_len] for w in waveforms]

            # 3) RMS-normalize each track for fair loudness comparison
            for i in range(len(waveforms)):
                rms = np.sqrt(np.mean(waveforms[i]**2))
                if rms > 1e-9:
                    waveforms[i] /= rms

            # 4) Compute absolute difference in time domain
            differences = np.abs(waveforms[0] - waveforms[1])

            current_step += 1
            if callback is not None:
                callback(current_step, "Calculated differences", total_steps)

            # --- SPECTROGRAM DIFFERENCE ---
            # Compute the magnitude STFT of each waveform
            f1, t1, Zxx1 = stft(waveforms[0], fs=sample_rate, nperseg=2048, noverlap=1024)
            f2, t2, Zxx2 = stft(waveforms[1], fs=sample_rate, nperseg=2048, noverlap=1024)

            # Convert to magnitude
            spec1 = np.abs(Zxx1)
            spec2 = np.abs(Zxx2)

            # For plotting, we might align their time axes
            min_t_len = min(spec1.shape[1], spec2.shape[1])
            spec1 = spec1[:, :min_t_len]
            spec2 = spec2[:, :min_t_len]
            t_combined = t1[:min_t_len]

            # Take difference or ratio in the frequency domain
            spec_diff = np.abs(spec1 - spec2)

            # 5) Plot results
            plt.figure(figsize=(15, 10))

            # Top row: waveforms
            plt.subplot(4, 1, 1)
            plt.plot(waveforms[0], label="Track 1 (RMS=1)", alpha=0.6)
            plt.plot(waveforms[1], label="Track 2 (RMS=1)", alpha=0.6, linestyle='dashed')
            plt.title("Waveforms (RMS-Normalized)")
            plt.legend()

            # 2nd row: absolute difference
            plt.subplot(4, 1, 2)
            plt.plot(differences, color='red', label="|Track1 - Track2|")
            plt.title("Time-Domain Differences")
            plt.legend()

            # 3rd row: spectrogram difference for Track 1
            plt.subplot(4, 1, 3)
            plt.title("Spectrogram Track 1 (Magnitude)")
            plt.pcolormesh(t_combined, f1, 20*np.log10(spec1 + 1e-9), cmap='viridis', shading='auto')
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Time [sec]")

            # 4th row: difference spectrogram
            plt.subplot(4, 1, 4)
            plt.title("Spectrogram Difference (|Spec1 - Spec2|)")
            plt.pcolormesh(t_combined, f1, 20*np.log10(spec_diff + 1e-9), cmap='magma', shading='auto')
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Time [sec]")

            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()

            current_step += 1
            if callback is not None:
                callback(current_step, f"Saved visualization to {output_file}", total_steps)
            project.add_output("comparison", output_file)
            pj_outputs.append(project)

        return pj_outputs

    def register_api_endpoint(self, api) -> Any:
        """
        Register FastAPI endpoint for audio comparison.
        
        Args:
            api: FastAPI application instance
            
        Returns:
            The registered endpoint route
        """
        from fastapi import HTTPException, Body
        
        # Create models for JSON API
        FileData, JsonRequest = self.create_json_models()

        @api.post("/api/v1/process/compare", tags=["Audio Processing"])
        async def process_compare_json(
            request: JsonRequest = Body(...)
        ):
            """
            Compare two audio files and generate visualization.
            
            This endpoint analyzes and compares two audio files, generating visualizations that show 
            time-domain waveforms, absolute difference, and spectrogram-based representations.
            
            ## Request Body
            
            ```json
            {
              "files": [
                {
                  "filename": "audio1.wav",
                  "content": "base64_encoded_file_content..."
                },
                {
                  "filename": "audio2.wav",
                  "content": "base64_encoded_file_content..."
                }
              ]
            }
            ```
            
            ## Parameters
            
            - **files**: Array of file objects, each containing:
              - **filename**: Name of the file (with extension)
              - **content**: Base64-encoded file content
              - Exactly 2 files must be provided for comparison
            
            ## Response
            
            ```json
            {
              "files": [
                {
                  "filename": "comparison.png",
                  "content": "base64_encoded_file_content..."
                }
              ]
            }
            ```
            
            The API returns an object containing the comparison visualization as a Base64-encoded PNG image.
            """
            # Check if exactly 2 files are provided
            if len(request.files) != 2:
                raise HTTPException(status_code=400, detail="Exactly 2 files must be provided for comparison")
                
            # Use the handle_json_request helper from BaseWrapper
            return self.handle_json_request(request, self.process_audio)

        return process_compare_json
        
    def test(self):
        """
        Test method for the Compare wrapper.
        This method creates simple test data and demonstrates the functionality of the wrapper.
        """
        print("Running Compare wrapper test...")
        
        # Create temporary directory for test
        import tempfile
        import shutil
        import numpy as np
        from scipy.io import wavfile
        from util.data_classes import ProjectFiles
        
        temp_dir = tempfile.mkdtemp(prefix="audiolab_compare_test_")
        print(f"Created temporary directory at: {temp_dir}")
        
        try:
            # Create two simple test signals
            duration = 3  # seconds
            sample_rate = 44100
            t = np.linspace(0, duration, sample_rate * duration)
            
            # Signal 1: Simple sine wave
            signal1 = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz A note
            
            # Signal 2: Slightly different sine wave with some noise
            signal2 = 0.5 * np.sin(2 * np.pi * 445 * t)  # 445 Hz, slightly off from A
            signal2 += 0.05 * np.random.normal(0, 1, len(signal2))  # Add noise
            
            # Save as WAV files
            file1_path = os.path.join(temp_dir, "test_signal1.wav")
            file2_path = os.path.join(temp_dir, "test_signal2.wav")
            
            wavfile.write(file1_path, sample_rate, signal1.astype(np.float32))
            wavfile.write(file2_path, sample_rate, signal2.astype(np.float32))
            
            print(f"Created test signals at: {file1_path} and {file2_path}")
            
            # Create ProjectFiles object for testing
            project = ProjectFiles(file1_path)
            project.project_dir = temp_dir
            project.add_raw_file(file2_path)
            
            # Define a simple callback function
            def test_callback(progress, message, total):
                print(f"Progress: {progress}/{total} - {message}")
            
            # Process the test data
            print("Processing test data...")
            results = self.process_audio([project], callback=test_callback)
            
            if not results:
                raise Exception("Test failed: No results returned")
            
            # Check if output file was created
            project_output = results[0]
            if len(project_output.last_outputs) == 0:
                raise Exception("Test failed: No output files created")
                
            output_file = project_output.last_outputs[0]
            if not os.path.exists(output_file):
                raise Exception(f"Test failed: Output file not found at {output_file}")
                
            print(f"Test successful! Output file created at: {output_file}")
            
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
        
        print("Compare wrapper test completed successfully!")
