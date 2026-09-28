"""
Automatic transcription for RVC V3 data preparation.

Uses Whisper for accurate transcription with word-level timestamps.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


class Transcriber:
    """
    Transcribe audio to text with word-level timestamps using Whisper.
    
    Provides high-quality transcription for lyric annotation.
    """
    
    def __init__(self, output_dir: str, model_size: str = "large-v3"):
        """
        Initialize the transcriber.
        
        Args:
            output_dir: Base directory for output (e.g., outputs/rvc_v3_data)
            model_size: Whisper model size (base, small, medium, large, large-v3)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_size = model_size
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Transcriber initialized (device: {self.device})")
    
    def _load_model(self):
        """Lazy load the Whisper model."""
        if self.model is not None:
            return
        
        try:
            import whisper
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe(
        self,
        audio_path: str,
        project_name: str,
        language: Optional[str] = None,
        word_timestamps: bool = True,
        overwrite_existing: bool = False,
    ) -> Optional[List[Dict]]:
        """
        Transcribe audio to text with timestamps.
        
        Args:
            audio_path: Path to the audio file
            project_name: Name of the project
            language: Language code (e.g., 'en', 'es') or None for auto-detect
            word_timestamps: Whether to generate word-level timestamps
        
        Returns:
            List of transcript segments with timing, or None on error
        """
        self._load_model()
        
        project_dir = self.output_dir / project_name
        lyrics_dir = project_dir / "lyrics"
        lyrics_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = lyrics_dir / "transcript.auto.json"
        if output_file.exists() and not overwrite_existing:
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("Skipping transcription; preserved existing lyrics: %s", output_file)
                return data.get("segments", [])
            except Exception:
                logger.warning("Could not read existing transcript %s; regenerating", output_file)

        try:
            logger.info(f"Transcribing {audio_path}")
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_path,
                language=language,
                word_timestamps=word_timestamps,
                verbose=False
            )
            
            # Extract segments with timestamps
            segments = []
            
            if word_timestamps and 'segments' in result:
                # Word-level timestamps
                for segment in result['segments']:
                    if 'words' in segment:
                        for word in segment['words']:
                            segments.append({
                                'start': word.get('start', 0.0),
                                'end': word.get('end', 0.0),
                                'text': word.get('word', '').strip(),
                                'confidence': word.get('probability', 1.0)
                            })
                    else:
                        # Fallback to segment-level
                        segments.append({
                            'start': segment.get('start', 0.0),
                            'end': segment.get('end', 0.0),
                            'text': segment.get('text', '').strip(),
                            'confidence': 1.0
                        })
            else:
                # Segment-level timestamps
                for segment in result.get('segments', []):
                    segments.append({
                        'start': segment.get('start', 0.0),
                        'end': segment.get('end', 0.0),
                        'text': segment.get('text', '').strip(),
                        'confidence': 1.0
                    })
            
            # Save transcript
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'language': result.get('language', 'unknown'),
                    'segments': segments,
                    'full_text': result.get('text', '')
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Transcription complete: {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None
    
    def transcribe_with_whisperx(
        self,
        audio_path: str,
        project_name: str,
        language: Optional[str] = None,
        overwrite_existing: bool = False,
    ) -> Optional[List[Dict]]:
        """
        Transcribe using WhisperX for better word alignment.
        
        WhisperX provides more accurate word-level timestamps using forced alignment.
        
        Args:
            audio_path: Path to the audio file
            project_name: Name of the project
            language: Language code or None for auto-detect
        
        Returns:
            List of transcript segments with timing, or None on error
        """
        try:
            import whisperx
            logger.info("Using WhisperX for enhanced alignment")
        except ImportError:
            logger.warning("WhisperX not available, falling back to standard Whisper")
            return self.transcribe(audio_path, project_name, language, overwrite_existing=overwrite_existing)
        
        project_dir = self.output_dir / project_name
        lyrics_dir = project_dir / "lyrics"
        lyrics_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = lyrics_dir / "transcript.auto.json"
        if output_file.exists() and not overwrite_existing:
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("Skipping WhisperX transcription; preserved existing lyrics: %s", output_file)
                return data.get("segments", [])
            except Exception:
                logger.warning("Could not read existing transcript %s; regenerating", output_file)

        try:
            logger.info(f"Transcribing with WhisperX: {audio_path}")
            
            # Load audio
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Transcribe with WhisperX
            model = whisperx.load_model(self.model_size, self.device)
            result = model.transcribe(audio, language=language)
            
            # Align whisper output
            model_a, metadata = whisperx.load_align_model(
                language_code=result.get("language", "en"),
                device=self.device
            )
            result_aligned = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self.device
            )
            
            # Extract word-level segments
            segments = []
            for segment in result_aligned.get("segments", []):
                if 'words' in segment:
                    for word in segment['words']:
                        segments.append({
                            'start': word.get('start', 0.0),
                            'end': word.get('end', 0.0),
                            'text': word.get('word', '').strip(),
                            'confidence': word.get('score', 1.0)
                        })
            
            # Save transcript
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'language': result.get('language', 'unknown'),
                    'segments': segments,
                    'full_text': ' '.join([s['text'] for s in segments])
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"WhisperX transcription complete: {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.error(f"WhisperX transcription failed: {e}")
            # Fallback to standard Whisper
            return self.transcribe(audio_path, project_name, language, overwrite_existing=overwrite_existing)
    
    def get_transcript(self, project_name: str) -> Optional[Dict]:
        """Load existing transcript for a project."""
        transcript_file = self.output_dir / project_name / "lyrics" / "transcript.auto.json"
        if not transcript_file.exists():
            legacy_file = self.output_dir / project_name / "lyrics" / "transcript.json"
            transcript_file = legacy_file if legacy_file.exists() else transcript_file
            if not transcript_file.exists():
                return None
        
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load transcript: {e}")
            return None

    def transcribe_file(
        self,
        audio_path: str,
        output_path: str,
        language: Optional[str] = None,
        word_timestamps: bool = True,
        overwrite_existing: bool = False,
    ) -> Optional[List[Dict]]:
        """
        Transcribe a single audio file and save to specified path.
        
        Args:
            audio_path: Path to the audio file
            output_path: Path to save transcript JSON
            language: Language code or None for auto-detect
            word_timestamps: Whether to generate word-level timestamps
        
        Returns:
            List of transcript segments, or None on error
        """
        self._load_model()
        
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        if out.exists() and not overwrite_existing:
            try:
                with open(out, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("Skipping transcription; preserved existing lyrics: %s", out)
                return data.get("segments", [])
            except Exception:
                logger.warning("Could not read existing transcript %s; regenerating", out)

        try:
            logger.info(f"Transcribing {audio_path}")
            result = self.model.transcribe(
                audio_path,
                language=language,
                word_timestamps=word_timestamps,
                verbose=False
            )
            
            segments = []
            if word_timestamps and 'segments' in result:
                for segment in result['segments']:
                    if 'words' in segment:
                        for word in segment['words']:
                            segments.append({
                                'start': word.get('start', 0.0),
                                'end': word.get('end', 0.0),
                                'text': word.get('word', '').strip(),
                                'confidence': word.get('probability', 1.0)
                            })
                    else:
                        segments.append({
                            'start': segment.get('start', 0.0),
                            'end': segment.get('end', 0.0),
                            'text': segment.get('text', '').strip(),
                            'confidence': 1.0
                        })
            else:
                for segment in result.get('segments', []):
                    segments.append({
                        'start': segment.get('start', 0.0),
                        'end': segment.get('end', 0.0),
                        'text': segment.get('text', '').strip(),
                        'confidence': 1.0
                    })
            
            data = {
                'language': result.get('language', 'unknown'),
                'segments': segments,
                'full_text': result.get('text', ''),
                'source': audio_path
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Transcription saved: {output_path} ({len(segments)} segments)")
            return segments
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            return None

    def transcribe_project_vocals(
        self,
        project_dir: str,
        gt_wavs_dir: Optional[str] = None,
        language: Optional[str] = None,
        overwrite_existing: bool = False,
    ) -> Dict[str, bool]:
        """
        Transcribe all vocal wavs in a project's 0_gt_wavs directory.
        Saves per-stem lyrics to project_dir/lyrics/<stem>.json.
        
        Args:
            project_dir: Project directory (e.g. outputs/voices/MyProject)
            gt_wavs_dir: Path to 0_gt_wavs, or None to use project_dir/0_gt_wavs
            language: Language code or None for auto-detect
        
        Returns:
            Dict mapping stem -> True if transcribed successfully, False otherwise
        """
        project_path = Path(project_dir)
        wavs_dir = Path(gt_wavs_dir) if gt_wavs_dir else project_path / "0_gt_wavs"
        lyrics_dir = project_path / "lyrics"
        lyrics_dir.mkdir(parents=True, exist_ok=True)
        
        if not wavs_dir.exists():
            logger.warning(f"gt_wavs dir not found: {wavs_dir}")
            return {}
        
        wav_files = sorted(wavs_dir.glob("*.wav"))
        results = {}
        
        for wav_path in wav_files:
            stem = wav_path.stem
            out_path = lyrics_dir / f"{stem}.json"
            
            segs = self.transcribe_file(
                str(wav_path),
                str(out_path),
                language=language,
                word_timestamps=True,
                overwrite_existing=overwrite_existing,
            )
            results[stem] = segs is not None
        
        n_ok = sum(1 for v in results.values() if v)
        logger.info(f"Transcribed {n_ok}/{len(results)} vocals in {project_dir}")
        return results

