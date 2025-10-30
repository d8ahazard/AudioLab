"""
RVC V3 Gradio UI Layout.

UI for data preparation, training, and inference with RVC V3.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import gradio as gr
import torch

from handlers.config import output_path
from modules.rvc_v3.data_prep import (
    SongDownloader, VocalSeparator, Transcriber, LyricEditor, Phonemizer
)
from modules.rvc_v3.training import FeatureExtractor, IndexBuilder, RVCV3Trainer
from modules.rvc_v3.configs import RVCV3Config, get_default_config
from modules.rvc_v3.inference import RVCV3Pipeline

logger = logging.getLogger(__name__)

# Global state
rvc_v3_data_dir = os.path.join(output_path, "rvc_v3_data")
os.makedirs(rvc_v3_data_dir, exist_ok=True)


def list_projects():
    """List all RVC V3 projects."""
    if not os.path.exists(rvc_v3_data_dir):
        return []
    
    projects = [d for d in os.listdir(rvc_v3_data_dir)
                if os.path.isdir(os.path.join(rvc_v3_data_dir, d))]
    return sorted(projects)


def refresh_project_list():
    """Refresh project dropdown."""
    projects = list_projects()
    return gr.Dropdown(choices=projects, value=projects[0] if projects else None)


def download_songs(urls: str, project_name: str, progress=gr.Progress()):
    """Download songs from multiple URLs."""
    try:
        if not urls or not urls.strip():
            return "✗ Please enter at least one URL", gr.Dropdown(choices=[])
        
        # Split by newlines and filter valid URLs
        import re
        url_list = re.split(r'\n', urls)
        valid_urls = [u.strip() for u in url_list if u.strip() and u.strip().startswith('http')]
        
        if not valid_urls:
            return "✗ No valid URLs found", gr.Dropdown(choices=[])
        
        progress(0, desc=f"Downloading {len(valid_urls)} songs...")
        
        downloader = SongDownloader(rvc_v3_data_dir)
        downloaded_songs = []
        
        for i, url in enumerate(valid_urls):
            try:
                progress((i / len(valid_urls)) * 0.8, desc=f"Downloading {i+1}/{len(valid_urls)}...")
                
                # Create unique project name for each song
                song_project = f"{project_name}_song_{i+1:02d}"
                
                audio_path, metadata = downloader.download(url, song_project)
                
                if audio_path:
                    title = metadata.get('title', f'song_{i+1}')
                    downloaded_songs.append({
                        'project': song_project,
                        'title': title,
                        'path': audio_path
                    })
                    logger.info(f"Downloaded: {title}")
                
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")
                continue
        
        progress(1.0, desc="Complete!")
        
        if downloaded_songs:
            # Update song selector dropdown
            song_choices = [f"{s['project']}: {s['title']}" for s in downloaded_songs]
            
            status = f"✓ Downloaded {len(downloaded_songs)}/{len(valid_urls)} songs:\n"
            status += "\n".join([f"  - {s['title']}" for s in downloaded_songs])
            
            return status, gr.Dropdown(choices=song_choices, value=song_choices[0] if song_choices else None)
        else:
            return f"✗ Failed to download any songs", gr.Dropdown(choices=[])
    
    except Exception as e:
        logger.error(f"Download error: {e}")
        return f"✗ Error: {str(e)}", gr.Dropdown(choices=[])


def get_song_project_name(song_selector: str) -> str:
    """Extract project name from song selector string."""
    if not song_selector:
        return ""
    # Format is "project_name: Song Title"
    return song_selector.split(':')[0].strip()


def separate_selected_song(song_selector: str, progress=gr.Progress()):
    """Separate vocals from selected song."""
    try:
        song_project = get_song_project_name(song_selector)
        if not song_project:
            return "✗ Please select a song"
        
        progress(0, desc="Initializing separator...")
        
        separator = VocalSeparator(rvc_v3_data_dir)
        
        # Find audio file
        project_dir = Path(rvc_v3_data_dir) / song_project / "raw"
        audio_files = list(project_dir.glob("*.wav"))
        
        if not audio_files:
            audio_files = list(project_dir.glob("*.mp3"))
        
        if not audio_files:
            return f"✗ No audio files found in {song_project}"
        
        audio_path = str(audio_files[0])
        
        progress(0.2, desc="Separating vocals...")
        vocals_path, instrumental_path = separator.separate_vocals(
            audio_path, song_project
        )
        
        if vocals_path:
            progress(1.0, desc="Complete!")
            return f"✓ Vocals separated for {song_selector}!\nVocals: {vocals_path}"
        else:
            return "✗ Separation failed"
    
    except Exception as e:
        logger.error(f"Separation error: {e}")
        return f"✗ Error: {str(e)}"


def transcribe_selected_song(song_selector: str, use_whisperx: bool, progress=gr.Progress()):
    """Transcribe vocals to text for selected song."""
    try:
        song_project = get_song_project_name(song_selector)
        if not song_project:
            return "✗ Please select a song", "", gr.Textbox()
        
        progress(0, desc="Initializing transcriber...")
        
        transcriber = Transcriber(rvc_v3_data_dir, model_size="large-v3")
        
        # Find vocal file
        vocals_path_obj = Path(rvc_v3_data_dir) / song_project / "vocals"
        vocal_files = list(vocals_path_obj.glob("*vocals*.wav"))
        
        if not vocal_files:
            vocal_files = list(vocals_path_obj.glob("*.wav"))
        
        if not vocal_files:
            return f"✗ No vocal files found for {song_selector}", "", gr.Textbox()
        
        vocal_path = str(vocal_files[0])
        
        progress(0.2, desc="Transcribing (this may take a while)...")
        
        if use_whisperx:
            segments = transcriber.transcribe_with_whisperx(vocal_path, song_project)
        else:
            segments = transcriber.transcribe(vocal_path, song_project)
        
        if segments:
            progress(1.0, desc="Complete!")
            
            # Format transcript for display
            transcript_text = "\n".join([
                f"{seg['start']:.2f}-{seg['end']:.2f}: {seg['text']}"
                for seg in segments
            ])
            
            # Create editable lyrics text
            lyrics_editable = "\n".join([seg['text'] for seg in segments])
            
            return (
                f"✓ Transcribed {len(segments)} segments for {song_selector}",
                transcript_text,
                gr.Textbox(value=lyrics_editable, interactive=True)
            )
        else:
            return "✗ Transcription failed", "", gr.Textbox()
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return f"✗ Error: {str(e)}", "", gr.Textbox()


def load_lyrics_for_editing(song_selector: str):
    """Load lyrics for editing and show transcript preview."""
    try:
        song_project = get_song_project_name(song_selector)
        if not song_project:
            return "", "No song selected"
        
        # Check for annotated lyrics first
        lyrics_file = Path(rvc_v3_data_dir) / song_project / "lyrics" / "annotated_lyrics.json"
        
        lyrics_text = ""
        transcript_preview = ""
        
        if lyrics_file.exists():
            # Load annotated lyrics
            import json
            with open(lyrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Format for editing: show with tags
            lines = []
            preview_lines = []
            
            for seg in data.get('segments', []):
                tags = seg.get('tags', [])
                text = seg['text']
                start = seg.get('start', 0)
                end = seg.get('end', 0)
                
                # Editable lyrics (with tags)
                if tags:
                    tag_str = ' '.join([f'[{tag}]' for tag in tags])
                    lines.append(f"{tag_str} {text}")
                else:
                    lines.append(text)
                
                # Preview with timestamps
                preview_lines.append(f"{start:.2f}-{end:.2f}: {text}")
            
            lyrics_text = "\n".join(lines)
            transcript_preview = "\n".join(preview_lines)
        
        else:
            # Check for raw transcript
            transcript_file = Path(rvc_v3_data_dir) / song_project / "lyrics" / "transcript.json"
            
            if transcript_file.exists():
                import json
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Editable lyrics (no tags yet)
                lyrics_text = "\n".join([seg['text'] for seg in data.get('segments', [])])
                
                # Preview with timestamps
                transcript_preview = "\n".join([
                    f"{seg['start']:.2f}-{seg['end']:.2f}: {seg['text']}"
                    for seg in data.get('segments', [])
                ])
            else:
                return "", f"✗ No lyrics found for {song_selector}. Please process songs first."
        
        return lyrics_text, transcript_preview
    
    except Exception as e:
        logger.error(f"Error loading lyrics: {e}")
        import traceback
        traceback.print_exc()
        return "", f"✗ Error: {str(e)}"


def save_edited_lyrics(song_selector: str, lyrics_text: str):
    """Save edited lyrics with tags."""
    try:
        song_project = get_song_project_name(song_selector)
        if not song_project:
            return "✗ Please select a song"
        
        if not lyrics_text or not lyrics_text.strip():
            return "✗ Lyrics are empty"
        
        # Parse lyrics text (each line can have tags)
        import re
        
        editor = LyricEditor(song_project, rvc_v3_data_dir)
        
        # Load existing transcript for timing
        transcript_file = Path(rvc_v3_data_dir) / song_project / "lyrics" / "transcript.json"
        
        if transcript_file.exists():
            import json
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            editor.load_from_transcript(transcript_data)
            
            # Parse edited lyrics and update segments
            lyrics_lines = [l.strip() for l in lyrics_text.split('\n') if l.strip()]
            
            # Match lines to segments (assuming same order)
            for i, line in enumerate(lyrics_lines):
                if i < len(editor.segments):
                    # Extract tags from line
                    tag_pattern = r'\[([^\]]+)\]'
                    tags = re.findall(tag_pattern, line)
                    text_no_tags = re.sub(tag_pattern, '', line).strip()
                    
                    # Update segment
                    editor.segments[i].text = text_no_tags
                    editor.segments[i].tags = tags
            
            # Save annotated lyrics
            saved_path = editor.save("annotated_lyrics.json")
            
            return f"✓ Lyrics saved to {saved_path}"
        else:
            return "✗ No transcript found. Please transcribe first to get timing information."
    
    except Exception as e:
        logger.error(f"Error saving lyrics: {e}")
        import traceback
        traceback.print_exc()
        return f"✗ Error: {str(e)}"


def get_all_songs_in_project(project_name: str):
    """Get all songs in a project."""
    if not project_name:
        return []
    
    project_base_dir = Path(rvc_v3_data_dir)
    
    # Find all song subprojects
    songs = []
    for item in project_base_dir.iterdir():
        if item.is_dir() and item.name.startswith(f"{project_name}_song_"):
            # Try to get title from metadata
            raw_dir = item / "raw"
            if raw_dir.exists():
                # Look for info.json
                json_files = list(raw_dir.glob("*.info.json"))
                if json_files:
                    try:
                        import json
                        with open(json_files[0], 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            title = metadata.get('title', item.name)
                    except:
                        title = item.name
                else:
                    title = item.name
                
                songs.append(f"{item.name}: {title}")
    
    return sorted(songs)


def download_and_process_all(urls: str, project_name: str, use_whisperx: bool, auto_process: bool, progress=gr.Progress()):
    """Download all songs and optionally process them."""
    try:
        if not urls or not urls.strip():
            return "✗ Please enter at least one URL", gr.Dropdown(choices=[])
        
        if not project_name or not project_name.strip():
            return "✗ Please enter a project name", gr.Dropdown(choices=[])
        
        # Split by newlines and filter valid URLs
        import re
        url_list = re.split(r'\n', urls)
        valid_urls = [u.strip() for u in url_list if u.strip() and u.strip().startswith('http')]
        
        if not valid_urls:
            return "✗ No valid URLs found", gr.Dropdown(choices=[])
        
        progress(0, desc=f"Downloading {len(valid_urls)} songs...")
        
        downloader = SongDownloader(rvc_v3_data_dir)
        separator = VocalSeparator(rvc_v3_data_dir)
        transcriber = Transcriber(rvc_v3_data_dir, model_size="large-v3")
        
        downloaded_songs = []
        results = []
        
        # Phase 1: Download all
        for i, url in enumerate(valid_urls):
            try:
                progress((i / len(valid_urls)) * 0.3, desc=f"Downloading {i+1}/{len(valid_urls)}...")
                
                # Create unique project name for each song
                song_project = f"{project_name}_song_{i+1:02d}"
                
                audio_path, metadata = downloader.download(url, song_project)
                
                if audio_path:
                    title = metadata.get('title', f'song_{i+1}')
                    downloaded_songs.append({
                        'project': song_project,
                        'title': title,
                        'path': audio_path
                    })
                    results.append(f"✓ Downloaded: {title}")
                    logger.info(f"Downloaded: {title}")
                else:
                    results.append(f"✗ Failed to download: {url}")
                
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")
                results.append(f"✗ Error downloading {url}: {str(e)}")
                continue
        
        if not downloaded_songs:
            return "\n".join(results), gr.Dropdown(choices=[])
        
        # Phase 2: Process all if auto_process is enabled
        if auto_process:
            progress(0.3, desc="Processing all songs...")
            
            for i, song_data in enumerate(downloaded_songs):
                song_project = song_data['project']
                title = song_data['title']
                
                # Separate vocals
                progress(0.3 + (i / len(downloaded_songs)) * 0.35, 
                        desc=f"Separating {i+1}/{len(downloaded_songs)}: {title}")
                
                vocals_path, _ = separator.separate_vocals(song_data['path'], song_project)
                
                if vocals_path:
                    results.append(f"✓ Separated: {title}")
                    
                    # Transcribe
                    progress(0.65 + (i / len(downloaded_songs)) * 0.35,
                            desc=f"Transcribing {i+1}/{len(downloaded_songs)}: {title}")
                    
                    if use_whisperx:
                        segments = transcriber.transcribe_with_whisperx(vocals_path, song_project)
                    else:
                        segments = transcriber.transcribe(vocals_path, song_project)
                    
                    if segments:
                        results.append(f"✓ Transcribed: {title} ({len(segments)} segments)")
                    else:
                        results.append(f"✗ Transcription failed: {title}")
                else:
                    results.append(f"✗ Separation failed: {title}")
        
        progress(1.0, desc="Complete!")
        
        # Update song selector
        song_choices = [f"{s['project']}: {s['title']}" for s in downloaded_songs]
        
        status = f"✓ Processed {len(downloaded_songs)} songs:\n\n" + "\n".join(results)
        
        return status, gr.Dropdown(choices=song_choices, value=song_choices[0] if song_choices else None)
    
    except Exception as e:
        logger.error(f"Download/process error: {e}")
        import traceback
        traceback.print_exc()
        return f"✗ Error: {str(e)}", gr.Dropdown(choices=[])


def extract_features(project_name: str, sample_rate: int, use_dual: bool, progress=gr.Progress()):
    """Extract training features from all songs in project."""
    try:
        progress(0, desc="Initializing feature extractor...")
        
        config = get_default_config(sample_rate)
        config.use_dual_encoder = use_dual
        
        extractor = FeatureExtractor(config, device="cuda", use_dual_encoder=use_dual)
        
        # Get all song projects
        songs = get_all_songs_in_project(project_name)
        
        if not songs:
            return "✗ No songs found in project"
        
        progress(0.1, desc=f"Extracting features from {len(songs)} songs...")
        
        total_processed = 0
        
        for i, song in enumerate(songs):
            song_project = get_song_project_name(song)
            project_dir = os.path.join(rvc_v3_data_dir, song_project)
            
            progress(0.1 + (i / len(songs)) * 0.8, desc=f"Processing {song}...")
            
            def callback(prog, msg, total):
                overall_prog = 0.1 + ((i + prog) / len(songs)) * 0.8
                progress(overall_prog, desc=f"{song}: {msg}")
            
            try:
                extractor.extract_project(project_dir, callback=callback)
                total_processed += 1
            except Exception as e:
                logger.error(f"Failed to extract features for {song}: {e}")
        
        progress(1.0, desc="Complete!")
        return f"✓ Feature extraction complete: {total_processed}/{len(songs)} songs processed"
    
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        import traceback
        traceback.print_exc()
        return f"✗ Error: {str(e)}"


def build_index(project_name: str, sample_rate: int, progress=gr.Progress()):
    """Build retrieval index from all songs in project."""
    try:
        progress(0, desc="Loading configuration...")
        
        config = get_default_config(sample_rate)
        feature_dim = config.get_content_feature_dim()
        
        # Collect features from all song projects
        songs = get_all_songs_in_project(project_name)
        
        if not songs:
            return "✗ No songs found in project"
        
        progress(0.1, desc=f"Collecting features from {len(songs)} songs...")
        
        all_features = []
        
        for i, song in enumerate(songs):
            song_project = get_song_project_name(song)
            features_dir = Path(rvc_v3_data_dir) / song_project / config.features_dir
            
            if not features_dir.exists():
                continue
            
            # Load all feature files from this song
            for feature_file in features_dir.glob("*.pt"):
                try:
                    features_data = torch.load(feature_file, map_location='cpu')
                    content_features = features_data['content']
                    
                    if isinstance(content_features, torch.Tensor):
                        content_features = content_features.numpy()
                    
                    all_features.append(content_features)
                except Exception as e:
                    logger.error(f"Failed to load {feature_file}: {e}")
        
        if not all_features:
            return "✗ No features found. Please extract features first."
        
        # Concatenate all features
        import numpy as np
        all_features = np.vstack(all_features)
        
        progress(0.5, desc=f"Building index from {all_features.shape[0]} feature vectors...")
        
        builder = IndexBuilder(
            feature_dim=feature_dim,
            index_type="IVF",
            n_clusters=min(256, all_features.shape[0] // 10),
            use_gpu=True
        )
        
        # Build index directly
        from modules.rvc_v3.models.retrieval import RetrievalIndex
        index = RetrievalIndex(feature_dim, index_type="IVF", use_gpu=True)
        index.build(all_features)
        
        # Save to main project directory
        index_path = Path(rvc_v3_data_dir) / project_name / "retrieval_index"
        index.save(str(index_path))
        
        progress(1.0, desc="Complete!")
        return f"✓ Index built: {index_path} ({all_features.shape[0]} vectors from {len(songs)} songs)"
    
    except Exception as e:
        logger.error(f"Index building error: {e}")
        import traceback
        traceback.print_exc()
        return f"✗ Error: {str(e)}"


def start_training(project_name: str, sample_rate: int, num_epochs: int, batch_size: int, progress=gr.Progress()):
    """Start V3 training."""
    try:
        progress(0, desc="Initializing training...")
        
        config = get_default_config(sample_rate)
        config.batch_size = batch_size
        
        project_dir = os.path.join(rvc_v3_data_dir, project_name)
        
        trainer = RVCV3Trainer(config, project_dir, device="cuda")
        
        # Initialize phonemizer
        phonemizer = Phonemizer(language='en-us')
        
        def callback(prog, msg, total):
            progress(prog, desc=msg)
        
        progress(0.1, desc="Training...")
        trainer.train(num_epochs, phonemizer, callback=callback)
        
        progress(1.0, desc="Training complete!")
        return "✓ Training complete"
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        return f"✗ Error: {str(e)}"


def list_checkpoints(project_name: str):
    """List available model checkpoints for a project."""
    if not project_name:
        return []
    
    checkpoint_dir = Path(rvc_v3_data_dir) / project_name / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    return [str(cp.name) for cp in sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)]


def list_indexes(project_name: str):
    """List available retrieval indexes for a project."""
    if not project_name:
        return []
    
    project_dir = Path(rvc_v3_data_dir) / project_name
    indexes = list(project_dir.glob("*.index"))
    return [str(idx.name) for idx in indexes]


def refresh_model_lists(project_name: str):
    """Refresh checkpoint and index dropdowns."""
    checkpoints = list_checkpoints(project_name)
    indexes = list_indexes(project_name)
    
    return (
        gr.Dropdown(choices=checkpoints, value=checkpoints[0] if checkpoints else None),
        gr.Dropdown(choices=indexes, value=indexes[0] if indexes else None)
    )


def convert_audio(
    project_name: str,
    checkpoint_name: str,
    index_name: str,
    input_audio_path: str,
    lyrics: str,
    index_rate: float,
    pitch_shift: int,
    sample_rate: int,
    progress=gr.Progress()
):
    """Convert audio using RVC V3."""
    try:
        if not input_audio_path:
            return None, "✗ Please upload input audio"
        
        if not project_name or not checkpoint_name:
            return None, "✗ Please select project and checkpoint"
        
        progress(0, desc="Loading model...")
        
        # Get paths
        project_dir = Path(rvc_v3_data_dir) / project_name
        checkpoint_path = project_dir / "checkpoints" / checkpoint_name
        
        if not checkpoint_path.exists():
            return None, f"✗ Checkpoint not found: {checkpoint_path}"
        
        # Load config
        config = get_default_config(sample_rate)
        
        progress(0.1, desc="Initializing pipeline...")
        
        # Initialize pipeline
        pipeline = RVCV3Pipeline(
            str(checkpoint_path),
            config,
            device="cuda"
        )
        
        # Load retrieval index if specified
        if index_name:
            index_path = project_dir / index_name
            if index_path.exists():
                progress(0.2, desc="Loading retrieval index...")
                pipeline.load_retrieval_index(str(index_path))
        
        progress(0.3, desc="Converting audio...")
        
        # Convert
        output_audio, sr = pipeline.convert(
            audio_path=input_audio_path,
            lyrics=lyrics if lyrics.strip() else None,
            index_rate=index_rate,
            pitch_shift=pitch_shift,
            speaker_id=0
        )
        
        progress(1.0, desc="Complete!")
        
        return (sr, output_audio), f"✓ Conversion complete! Sample rate: {sr} Hz"
    
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        import traceback
        traceback.print_exc()
        return None, f"✗ Error: {str(e)}"


def create_rvc_v3_layout():
    """Create RVC V3 Gradio layout."""
    
    with gr.Blocks() as rvc_v3_ui:
        gr.Markdown("# RVC V3: Next-Generation Voice Conversion")
        gr.Markdown("Text-conditioned voice conversion with enhanced fidelity and stereo support")
        
        # Project Management
        with gr.Row():
            project_dropdown = gr.Dropdown(
                label="Select Project",
                choices=list_projects(),
                value=list_projects()[0] if list_projects() else None
            )
            refresh_btn = gr.Button("🔄 Refresh", scale=0)
            new_project_name = gr.Textbox(label="New Project Name", placeholder="my_voice_model")
        
        refresh_btn.click(refresh_project_list, outputs=[project_dropdown])
        
        # Tabs for different stages
        with gr.Tabs():
            # Data Preparation Tab
            with gr.Tab("📥 Data Preparation"):
                gr.Markdown("### Step 1: Download & Process Songs")
                gr.Markdown("*Enter one or more YouTube/audio URLs (one per line)*")
                
                song_urls = gr.Textbox(
                    label="Audio URLs",
                    placeholder="https://youtube.com/watch?v=...\nhttps://youtube.com/watch?v=...\n(one per line)",
                    lines=5
                )
                
                with gr.Row():
                    use_whisperx = gr.Checkbox(label="Use WhisperX (better alignment)", value=False)
                    auto_process = gr.Checkbox(label="Auto-process (separate + transcribe)", value=True)
                
                download_process_btn = gr.Button("Download & Process All Songs", variant="primary", size="lg")
                processing_output = gr.Textbox(label="Processing Status", lines=8)
                
                gr.Markdown("### Step 2: Review & Edit Lyrics")
                gr.Markdown("*Select each song to review transcription accuracy and add style tags*")
                
                with gr.Row():
                    song_selector = gr.Dropdown(
                        label="Select Song to Edit",
                        choices=[],
                        interactive=True
                    )
                    refresh_songs_btn = gr.Button("🔄 Refresh", scale=0)
                
                with gr.Row():
                    transcribe_output = gr.Textbox(label="Transcript Preview (with timestamps)", lines=8)
                
                lyrics_editor = gr.Textbox(
                    label="Lyrics Editor (add tags like: [clean] your lyrics here)",
                    placeholder="Select a song above to load and edit its lyrics...",
                    lines=15,
                    interactive=True
                )
                
                with gr.Row():
                    save_lyrics_btn = gr.Button("Save Edited Lyrics", variant="primary")
                    lyrics_status = gr.Textbox(label="Save Status", lines=1, scale=2)
                
                gr.Markdown("""
                **Available Style Tags**: `[clean]`, `[raspy]`, `[breathy]`, `[distorted]`, `[nasal]`, 
                `[soft]`, `[belted]`, `[whisper]`, `[growl]`, `[vibrato]`, `[straight]`, 
                `[head_voice]`, `[chest_voice]`, `[falsetto]` or create your own!
                
                **Editing Tips**:
                - Edit the text to fix any transcription errors
                - Add style tags at the beginning of lines where vocal style changes
                - Example: `[clean] Never gonna give you up` or `[raspy] Never gonna let you down`
                - Save after editing to preserve your changes
                """)
                
                # Wire up callbacks
                download_process_btn.click(
                    download_and_process_all,
                    inputs=[song_urls, new_project_name, use_whisperx, auto_process],
                    outputs=[processing_output, song_selector]
                )
                
                refresh_songs_btn.click(
                    lambda pn: gr.Dropdown(choices=get_all_songs_in_project(pn)),
                    inputs=[project_dropdown],
                    outputs=[song_selector]
                )
                
                # Auto-load lyrics when song is selected
                song_selector.change(
                    load_lyrics_for_editing,
                    inputs=[song_selector],
                    outputs=[lyrics_editor, transcribe_output]
                )
                
                save_lyrics_btn.click(
                    save_edited_lyrics,
                    inputs=[song_selector, lyrics_editor],
                    outputs=[lyrics_status]
                )
            
            # Training Tab
            with gr.Tab("🎓 Training"):
                gr.Markdown("### Feature Extraction")
                with gr.Row():
                    sample_rate_train = gr.Radio(
                        choices=[32000, 40000, 44100, 48000],
                        value=44100,
                        label="Sample Rate"
                    )
                    use_dual_encoder = gr.Checkbox(label="Use Dual Encoder (HuBERT + Whisper)", value=True)
                
                extract_btn = gr.Button("Extract Features", variant="primary")
                extract_output = gr.Textbox(label="Extraction Status", lines=2)
                
                gr.Markdown("### Build Retrieval Index")
                build_index_btn = gr.Button("Build Index", variant="primary")
                index_output = gr.Textbox(label="Index Status", lines=2)
                
                gr.Markdown("### Start Training")
                with gr.Row():
                    num_epochs = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Epochs")
                    batch_size = gr.Slider(minimum=1, maximum=16, value=4, step=1, label="Batch Size")
                
                train_btn = gr.Button("Start Training", variant="primary")
                train_output = gr.Textbox(label="Training Status", lines=3)
                
                # Wire up callbacks
                extract_btn.click(
                    extract_features,
                    inputs=[project_dropdown, sample_rate_train, use_dual_encoder],
                    outputs=[extract_output]
                )
                
                build_index_btn.click(
                    build_index,
                    inputs=[project_dropdown, sample_rate_train],
                    outputs=[index_output]
                )
                
                train_btn.click(
                    start_training,
                    inputs=[project_dropdown, sample_rate_train, num_epochs, batch_size],
                    outputs=[train_output]
                )
            
            # Inference Tab
            with gr.Tab("🎤 Inference"):
                gr.Markdown("### Convert Audio with RVC V3")
                
                with gr.Row():
                    model_checkpoint = gr.Dropdown(label="Model Checkpoint", choices=[])
                    index_file = gr.Dropdown(label="Retrieval Index", choices=[])
                    refresh_models_btn = gr.Button("🔄 Refresh Models", scale=0)
                
                input_audio = gr.Audio(label="Input Audio", type="filepath")
                
                with gr.Row():
                    lyrics_input = gr.Textbox(
                        label="Lyrics (with optional [tags])",
                        placeholder="[clean] Your lyrics here...",
                        lines=5
                    )
                
                with gr.Row():
                    sample_rate_inference = gr.Radio(
                        choices=[32000, 40000, 44100, 48000],
                        value=44100,
                        label="Sample Rate"
                    )
                    index_rate = gr.Slider(0, 1, value=0.75, label="Index Rate")
                    pitch_shift = gr.Slider(-12, 12, value=0, step=1, label="Pitch Shift (semitones)")
                
                convert_btn = gr.Button("Convert", variant="primary")
                output_audio = gr.Audio(label="Output Audio", type="numpy")
                convert_status = gr.Textbox(label="Status")
                
                gr.Markdown("""
                ### Style Tags
                Use tags in square brackets to control vocal style:
                - `[clean]` - Clean, clear vocals
                - `[raspy]` - Raspy, rough vocals
                - `[breathy]` - Breathy, soft vocals
                - `[belted]` - Powerful, belted vocals
                - `[whisper]` - Whispered vocals
                - Custom tags can be used if trained on tagged data
                """)
                
                # Wire up inference callbacks
                refresh_models_btn.click(
                    refresh_model_lists,
                    inputs=[project_dropdown],
                    outputs=[model_checkpoint, index_file]
                )
                
                # Auto-refresh models when project changes
                project_dropdown.change(
                    refresh_model_lists,
                    inputs=[project_dropdown],
                    outputs=[model_checkpoint, index_file]
                )
                
                convert_btn.click(
                    convert_audio,
                    inputs=[
                        project_dropdown,
                        model_checkpoint,
                        index_file,
                        input_audio,
                        lyrics_input,
                        index_rate,
                        pitch_shift,
                        sample_rate_inference
                    ],
                    outputs=[output_audio, convert_status]
                )
    
    return rvc_v3_ui


# Create the layout
def create_rvc_v3_tab():
    """Create RVC V3 tab for main UI."""
    return create_rvc_v3_layout()

