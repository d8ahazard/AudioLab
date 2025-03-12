import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
import torch
from handlers.config import app_path

import gradio as gr
from gradio_vistimeline import VisTimeline, VisTimelineData

from handlers.args import ArgHandler
from handlers.config import model_path
from modules.yue.source.infer import GenerationToken, GenerationParams, Generator, Stage1Config, Stage2Config
from modules.yue.source.song import Song, GenerationCache, parse_lyrics

logger = logging.getLogger(__name__)

# Global state variables
SEND_TO_PROCESS_BUTTON: gr.Button = None
OUTPUT_MIX: gr.Audio = None

# Language mapping for selecting the correct Stage 1 model
STAGE1_MODELS = {
    "English": {
        "cot": "m-a-p/YuE-s1-7B-anneal-en-cot",
        "icl": "m-a-p/YuE-s1-7B-anneal-en-icl"
    },
    "Mandarin/Cantonese": {
        "cot": "m-a-p/YuE-s1-7B-anneal-zh-cot",
        "icl": "m-a-p/YuE-s1-7B-anneal-zh-icl"
    },
    "Japanese/Korean": {
        "cot": "m-a-p/YuE-s1-7B-anneal-jp-kr-cot",
        "icl": "m-a-p/YuE-s1-7B-anneal-jp-kr-icl"
    }
}

# Helper Enums
class EnumHelper(Enum):
    def __str__(self):
        if isinstance(self.value, tuple):
            return self.value[1]
        return self.name

    @classmethod
    def from_string(cls, enum_str):
        for e in cls:
            if e.name == enum_str:
                return e
            if isinstance(e.value, tuple) and e.value[1] == enum_str:
                return e
        raise ValueError(f'{cls.__name__} has no enum matching "{enum_str}"')

class GenerationMode(EnumHelper):
    Full = 0
    Continue = 1

class GenerationFormat(EnumHelper):
    Mp3 = 0
    Wav = 1

class GenerationStage(EnumHelper):
    Stage1 = (0, 'Stage 1')
    Stage2 = (1, 'Stage 2')

class GenerationStageMode(EnumHelper):
    Stage1 = (0, 'Stage 1')
    Stage2 = (1, 'Stage 2')
    Stage1And2 = (2, 'Stage 1+2')
    Stage1Post = (3, 'Stage 1 cache only')
    Stage2Post = (4, 'Stage 2 cache only')

class AudioPromptMode(EnumHelper):
    Off=(0, "Off")
    SingleTrack=(1, "Single Track")
    DualTrack=(2, "Dual Track")

# Helper functions
def tokens_to_ms(nr_tokens: int):
    return nr_tokens * 1000 // 50

def ms_to_tokens(milliseconds: int):
    return (milliseconds * 50) // 1000

def seconds_to_tokens(seconds: int):
    return seconds * 50

def tokens_to_seconds(tokens: int):
    return tokens // 50

def date_to_milliseconds(date):
    try:
        date = int(date)
    except ValueError:
        pass
    if isinstance(date, int):
        return date
    elif isinstance(date, str):
        dt = datetime.fromisoformat(date.replace("Z", "+00:00"))
        epoch = datetime(1970, 1, 1, tzinfo=dt.tzinfo)
        return int((dt - epoch).total_seconds() * 1000)
    else:
        return 0

def load_and_process_genres(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    categories = ['genre', 'timbre', 'gender', 'mood', 'instrument']
    all_items = [item.strip() for category in categories for item in data.get(category, [])]
    
    unique_items = {}
    for item in all_items:
        key = item.lower()
        if key not in unique_items and item:
            unique_items[key] = item
    
    return sorted(unique_items.values(), key=lambda x: x.lower())

@dataclass
class AudioPlayer:
    column: gr.Column
    accept_button: gr.Button
    reject_button: gr.Button
    audio_file: gr.File

class YueUI:
    MaxBatches = 10
    CacheTimelineGroup = 0
    SongTimelineGroup = 1
    MinTimelineBlockMs = 20
    DefaultStage1Model = "YuE-s1-7B-anneal-en-cot-exl2"
    DefaultStage1CacheMode = "Q4"
    DefaultStage2Model = "YuE-s2-1B-general-exl2"
    DefaultStage2CacheMode = "FP16"

    def __init__(self):
        self._timeline_groups = [
            {"id": 0, "content": "Cache"},
            {"id": 1, "content": "Segments"},
        ]
        self._players = []
        self._component_serializers = {}
        self._generation_token = None
        self._generation_outputs = None
        self._selected_timeline_items = []
        self._generation_cache = None
        self._arg_handler = None

    def render(self, arg_handler: ArgHandler):
        global SEND_TO_PROCESS_BUTTON, OUTPUT_MIX
        self._arg_handler = arg_handler
        with gr.Blocks() as app:
            # Load required scripts and styles
            gr.HTML(value="""
                <link rel="stylesheet" href="/yue/scripts/style.css">
                <script type="module">
                    import WaveSurfer from '/yue/scripts/wavesurfer.esm.js';
                    window.WaveSurfer = WaveSurfer;
                </script>
                <script src="/yue/scripts/utils.js"></script>
                <script src="/yue/scripts/audioplayer.js"></script>
            """)
            
            gr.Markdown("## YuE Music Generation")
            
            # Create main UI components
            with gr.Row():
                with gr.Column():
                    self._create_settings_column()
                with gr.Column():
                    self._create_input_column()
                with gr.Column():
                    self._create_output_column()
            
            # Initialize state
            self._generation_token = gr.State()
            self._generation_outputs = gr.State()
            self._selected_timeline_items = gr.State([])
            self._generation_cache = gr.State(GenerationCache(Song.NrStages))
            
            # Set up event handlers
            self._setup_event_handlers()
            
            # Store global references
            SEND_TO_PROCESS_BUTTON = self._send_to_process_button
            OUTPUT_MIX = self._output_mix

        return app

    def _create_settings_column(self):
        gr.Markdown("### 🔧 Settings")
        
        self._model_language = gr.Dropdown(
            ["English", "Mandarin/Cantonese", "Japanese/Korean"],
            value="English",
            label="Model Language",
            elem_classes="hintitem",
            elem_id="yue_model_language"
        )
        
        self._max_new_tokens = gr.Slider(
            500, 5000, value=3000, step=100,
            label="Max New Tokens",
            elem_classes="hintitem",
            elem_id="yue_max_new_tokens"
        )
        
        self._run_n_segments = gr.Slider(
            1, 10, value=2, step=1,
            label="Run N Segments",
            elem_classes="hintitem",
            elem_id="yue_run_n_segments"
        )
        
        self._stage2_batch_size = gr.Slider(
            1, 8, value=4, step=1,
            label="Stage 2 Batch Size",
            elem_classes="hintitem",
            elem_id="yue_stage2_batch_size"
        )
        
        self._keep_intermediate = gr.Checkbox(
            label="Keep Intermediate Files",
            value=False,
            elem_classes="hintitem",
            elem_id="yue_keep_intermediate"
        )
        
        self._disable_offload_model = gr.Checkbox(
            label="Disable Model Offloading",
            value=False,
            elem_classes="hintitem",
            elem_id="yue_disable_offload_model"
        )
        
        self._rescale = gr.Checkbox(
            label="Rescale Output",
            elem_classes="hintitem",
            elem_id="yue_rescale"
        )
        
        self._cuda_idx = gr.Number(
            value=0,
            label="CUDA Index",
            elem_classes="hintitem",
            elem_id="yue_cuda_idx"
        )
        
        self._seed = gr.Slider(
            value=-1,
            label="Seed",
            minimum=-1,
            maximum=4294967295,
            step=1,
            elem_classes="hintitem",
            elem_id="yue_seed"
        )

        # Model settings
        with gr.Accordion(label="Model Settings", open=False):
            self._stage1_model = gr.Dropdown(
                label="Stage1 Model",
                choices=self._get_model_choices(),
                value=self.DefaultStage1Model,
                elem_classes="hintitem",
                elem_id="yue_stage1_model"
            )

            self._stage1_model_cache_mode = gr.Dropdown(
                choices=["FP16", "Q8", "Q6", "Q4"],
                label="Stage 1 Cache Mode",
                value=self.DefaultStage1CacheMode,
                elem_classes="hintitem",
                elem_id="yue_stage1_cache_mode"
            )

            self._stage1_model_cache_size = gr.Number(
                label="Stage 1 Cache Size",
                value=3500,
                precision=0,
                info="The cache size used in Stage 1. This is the max context length.",
                elem_classes="hintitem",
                elem_id="yue_stage1_model_cache_size"
            )

            self._stage2_model = gr.Dropdown(
                label="Stage2 Model",
                choices=self._get_model_choices(),
                value=self.DefaultStage2Model,
                elem_classes="hintitem",
                elem_id="yue_stage2_model"
            )

            self._stage2_model_cache_mode = gr.Dropdown(
                choices=["FP16", "Q8", "Q6", "Q4"],
                label="Stage 2 Cache Mode",
                value=self.DefaultStage2CacheMode,
                elem_classes="hintitem",
                elem_id="yue_stage2_cache_mode"
            )

            self._stage2_model_cache_size = gr.Number(
                label="Stage 2 Cache Size",
                value=6000,
                precision=0,
                info="The cache size used in Stage 2.",
                elem_classes="hintitem",
                elem_id="yue_stage2_model_cache_size"
            )

        # Generation settings
        with gr.Accordion(label="Generation Settings", open=False):
            self._generation_stage_mode = gr.Radio(
                label="Stage configuration",
                choices=[str(mode) for mode in GenerationStageMode],
                value=str(GenerationStageMode.Stage1And2),
                elem_classes="hintitem",
                elem_id="yue_generation_stage_mode"
            )

            self._generation_mode = gr.Radio(
                label="Generation mode",
                choices=[str(mode) for mode in GenerationMode],
                value=str(GenerationMode.Continue),
                elem_classes="hintitem",
                elem_id="yue_generation_mode"
            )

            self._generation_format = gr.Radio(
                label="Format",
                choices=[str(fmt) for fmt in GenerationFormat],
                value=str(GenerationFormat.Mp3),
                elem_classes="hintitem",
                elem_id="yue_generation_format"
            )

            self._generation_length = gr.Slider(
                label="Generation length",
                value=5,
                minimum=1,
                maximum=60,
                step=1,
                elem_classes="hintitem",
                elem_id="yue_generation_length"
            )

            # Update visibility based on generation mode
            self._generation_mode.change(
                fn=lambda mode: gr.update(visible=mode == str(GenerationMode.Continue)),
                inputs=[self._generation_mode],
                outputs=[self._generation_length]
            )

        # Additional generation settings that are missing from the current implementation
        with gr.Accordion(label="Advanced Generation Settings", open=False):
            self._generation_stage1_cfg_scale = gr.Slider(
                label="CFG Scale",
                value=1.5,
                minimum=0.05,
                maximum=2.5,
                step=0.05,
                elem_classes="hintitem",
                elem_id="yue_generation_stage1_cfg_scale"
            )
            
            self._generation_stage1_top_p = gr.Slider(
                label="Top P",
                value=0.93,
                minimum=0,
                maximum=1,
                step=0.01,
                elem_classes="hintitem",
                elem_id="yue_generation_stage1_top_p"
            )
            
            self._generation_stage1_temperature = gr.Slider(
                label="Temperature",
                value=1,
                minimum=0.01,
                maximum=5,
                step=0.01,
                elem_classes="hintitem",
                elem_id="yue_generation_stage1_temperature"
            )
            
            self._generation_randomize_seed = gr.Checkbox(
                label="Randomize Seed",
                value=True,
                elem_classes="hintitem",
                elem_id="yue_generation_randomize_seed"
            )
            
            self._generation_batches = gr.Slider(
                label="Batches",
                value=1,
                minimum=1,
                maximum=YueUI.MaxBatches,
                step=1,
                elem_classes="hintitem",
                elem_id="yue_generation_batches"
            )

    def _create_input_column(self):
        gr.Markdown("### 🎤 Inputs")
        
        # Genre selection
        genre_path = os.path.join(app_path, "modules", "yue", "top_200_tags.json")
        genres = load_and_process_genres(genre_path)
        self._genre_selection = gr.Dropdown(
            label="Select Music Genres",
            choices=genres,
            multiselect=True,
            allow_custom_value=True,
            max_choices=50,
            elem_classes="hintitem",
            elem_id="yue_genre_selection"
        )
        
        # Lyrics input
        self._lyrics_text = gr.Textbox(
            label="Lyrics",
            lines=10,
            placeholder="Enter structured lyrics here... (Use [verse], [chorus] labels)",
            elem_classes="hintitem",
            elem_id="yue_lyrics_text"
        )

        # System prompt
        self._system_prompt = gr.Textbox(
            label="Stage 1 system prompt",
            value="Generate music from the given lyrics segment by segment.",
            elem_classes="hintitem",
            elem_id="yue_system_prompt"
        )

        # Default segment length
        self._default_segment_length = gr.Number(
            label="Segment length",
            value=30,
            info="Preferred song segment length in seconds",
            elem_classes="hintitem",
            elem_id="yue_default_segment_length"
        )

        # Audio prompt settings
        with gr.Accordion("Audio Prompt Settings", open=False):
            self._audio_prompt_mode = gr.Radio(
                label="Audio prompt mode",
                choices=[str(mode) for mode in AudioPromptMode],
                value=str(AudioPromptMode.Off),
                elem_classes="hintitem",
                elem_id="yue_audio_prompt_mode"
            )

            with gr.Column(visible=False) as self._audio_prompt_settings:
                self._audio_prompt_file = gr.File(
                    label="Upload Audio File",
                    file_types=["audio"],
                    elem_classes="hintitem",
                    elem_id="yue_audio_prompt_file"
                )
                
                self._vocal_track_prompt_file = gr.File(
                    label="Upload Vocal Track",
                    file_types=["audio"],
                    visible=False,
                    elem_classes="hintitem",
                    elem_id="yue_vocal_track_prompt_file"
                )
                
                self._instrumental_track_prompt_file = gr.File(
                    label="Upload Instrumental Track",
                    file_types=["audio"],
                    visible=False,
                    elem_classes="hintitem",
                    elem_id="yue_instrumental_track_prompt_file"
                )

                self._audio_prompt_start_time = gr.Number(
                    label="Prompt Start Time (s)",
                    value=0,
                    elem_classes="hintitem",
                    elem_id="yue_audio_prompt_start_time"
                )

                self._audio_prompt_end_time = gr.Number(
                    label="Prompt End Time (s)",
                    value=30,
                    elem_classes="hintitem",
                    elem_id="yue_audio_prompt_end_time"
                )

            # Update audio prompt visibility
            self._audio_prompt_mode.change(
                fn=self._on_audio_prompt_mode_change,
                inputs=[self._audio_prompt_mode],
                outputs=[
                    self._audio_prompt_settings,
                    self._audio_prompt_file,
                    self._vocal_track_prompt_file,
                    self._instrumental_track_prompt_file
                ]
            )

    def _create_output_column(self):
        gr.Markdown("### 🎶 Outputs")
        self._generation_progress = gr.Textbox(label="Generation Progress", max_lines=10)
        with gr.Row():
            self._start_button = gr.Button("Generate Music", variant="primary")
            self._send_to_process_button = gr.Button("Send to Process", variant="secondary")
            self._stop_button = gr.Button("Stop", variant="secondary")
        
        self._output_info = gr.Textbox(label="Output Info", max_lines=10)
        self._output_mix = gr.Audio(label="Final Mix", type="filepath")
        
        # Create timeline
        self._create_timeline()
        
        # Create audio players
        self._create_audio_players(YueUI.MaxBatches)

    def _create_timeline(self):
        self._timeline = VisTimeline(
            label="Generated song",
            value={"groups": self._timeline_groups},
            options={
                "moment": "+00:00",
                "showCurrentTime": False,
                "editable": {
                    "add": False,
                    "remove": False,
                    "updateGroup": False,
                    "updateTime": True,
                    "overrideItems": False,
                },
                "multiselect": True,
                "stack": False,
            }
        )

        # Add timeline control buttons
        with gr.Row(visible=False) as self._timeline_options_row:
            self._timeline_toggle_mute_button = gr.Button("Toggle mute segment(s)")
            with gr.Row(visible=True) as self._timeline_extra_options_row:
                self._timeline_split_button = gr.Button("Split segment")
                self._timeline_remove_button = gr.Button("Remove segment")

    def _setup_event_handlers(self):
        """Set up all event handlers for the UI"""
        # Generation events
        self._start_button.click(
            fn=self._on_generate_start,
            inputs=[],
            outputs=[self._generation_token, self._start_button, self._stop_button, self._generation_progress]
        ).then(
            fn=self._on_generate_click,
            inputs=[self._generation_token, self._generation_cache],
            outputs=[self._output_info, self._generation_outputs]
        ).then(
            fn=self._on_generate_complete,
            outputs=[self._generation_token, self._start_button, self._stop_button, self._generation_progress]
        )

        self._stop_button.click(
            fn=self._on_generate_stop,
            inputs=[self._generation_token]
        )

        # Timeline events
        self._timeline.item_select(
            fn=self._on_timeline_select,
            inputs=[self._timeline],
            outputs=[self._selected_timeline_items]
        )

        self._timeline.change(
            fn=self._update_timeline_from_event,
            inputs=[self._timeline, self._generation_cache],
            outputs=[self._generation_cache]
        ).then(
            fn=self._update_timeline,
            inputs=[self._generation_cache],
            outputs=[self._timeline]
        )

        # Timeline button events
        self._timeline_split_button.click(
            fn=self._split_segment,
            inputs=[self._lyrics_text, self._generation_cache],
            outputs=[self._generation_cache]
        ).then(
            fn=lambda: [],
            outputs=[self._selected_timeline_items]
        ).then(
            fn=self._update_timeline,
            inputs=[self._generation_cache],
            outputs=[self._timeline]
        )

        self._timeline_remove_button.click(
            fn=self._remove_segment,
            inputs=[self._generation_cache],
            outputs=[self._generation_cache]
        ).then(
            fn=lambda: [],
            outputs=[self._selected_timeline_items]
        ).then(
            fn=self._update_timeline,
            inputs=[self._generation_cache],
            outputs=[self._timeline]
        )

        self._timeline_toggle_mute_button.click(
            fn=self._toggle_mute_segments,
            inputs=[self._selected_timeline_items, self._generation_cache],
            outputs=[self._generation_cache]
        ).then(
            fn=self._update_timeline,
            inputs=[self._generation_cache],
            outputs=[self._timeline]
        )

        # Cache events
        self._generation_cache.change(
            fn=self._update_timeline,
            inputs=[self._generation_cache],
            outputs=[self._timeline]
        )

        # Selected items visibility
        self._selected_timeline_items.change(
            fn=self._update_timeline_options_visibility,
            inputs=[self._selected_timeline_items, self._generation_cache],
            outputs=[
                self._timeline_options_row,
                self._timeline_extra_options_row
            ]
        )

    def listen(self):
        """Set up event handlers for integration with main app"""
        process_inputs = self._arg_handler.get_element("main", "process_inputs")
        if process_inputs:
            self._send_to_process_button.click(
                fn=self._send_to_process,
                inputs=[self._output_mix, process_inputs],
                outputs=[process_inputs]
            )

    def _send_to_process(self, output_mix, process_inputs):
        if not output_mix or not os.path.exists(output_mix):
            return gr.update()
        if output_mix in process_inputs:
            return gr.update()
        process_inputs.append(output_mix)
        return gr.update(value=process_inputs)

    def _on_generate_click(self, token: GenerationToken, cache: GenerationCache):
        """Handle generation button click"""
        try:
            # Initialize generation token if needed
            if not token:
                token = GenerationToken()
                token.start_generation()

            # Get current settings
            prompt_mode = AudioPromptMode.from_string(str(self._audio_prompt_mode.value))
            generation_mode = GenerationMode.from_string(str(self._generation_mode.value))
            generation_stage_mode = GenerationStageMode.from_string(str(self._generation_stage_mode.value))
            output_format = GenerationFormat.from_string(str(self._generation_format.value))

            # Determine which stages to run
            generation_stages = set()
            output_stage = None

            match generation_stage_mode:
                case GenerationStageMode.Stage1:
                    generation_stages.add(GenerationStage.Stage1)
                    output_stage = GenerationStage.Stage1
                case GenerationStageMode.Stage2:
                    generation_stages.add(GenerationStage.Stage2)
                    output_stage = GenerationStage.Stage2
                case GenerationStageMode.Stage1And2:
                    generation_stages.add(GenerationStage.Stage1)
                    generation_stages.add(GenerationStage.Stage2)
                    output_stage = GenerationStage.Stage2
                case GenerationStageMode.Stage1Post:
                    output_stage = GenerationStage.Stage1
                case GenerationStageMode.Stage2Post:
                    output_stage = GenerationStage.Stage2

            # Set up generation parameters
            params = GenerationParams(
                token=token,
                max_new_tokens=seconds_to_tokens(self._generation_length.value) if generation_mode == GenerationMode.Continue else None,
                resume=generation_mode == GenerationMode.Continue,
                use_audio_prompt=prompt_mode == AudioPromptMode.SingleTrack,
                use_dual_tracks_prompt=prompt_mode == AudioPromptMode.DualTrack,
                prompt_start_time=self._audio_prompt_start_time.value,
                prompt_end_time=self._audio_prompt_end_time.value,
                audio_prompt_path=self._audio_prompt_file.value,
                instrumental_track_prompt_path=self._instrumental_track_prompt_file.value,
                vocal_track_prompt_path=self._vocal_track_prompt_file.value,
                stage1_guidance_scale=self._generation_stage1_cfg_scale.value,
                stage1_top_p=self._generation_stage1_top_p.value,
                stage1_temperature=self._generation_stage1_temperature.value,
                stage1_repetition_penalty=self._generation_repetition_penalty.value,
                rescale=self._rescale.value,
                hq_audio=output_format == GenerationFormat.Wav,
                output_dir=os.path.join(model_path, "outputs"),
            )

            # Configure models
            stage1_config = Stage1Config(
                model_path=self._stage1_model.value,
                cache_mode=self._stage1_model_cache_mode.value,
                cache_size=int(self._stage1_model_cache_size.value),
            )

            stage2_config = Stage2Config(
                model_path=self._stage2_model.value,
                cache_mode=self._stage2_model_cache_mode.value,
                cache_size=int(self._stage2_model_cache_size.value),
            )

            # Initialize generator
            generator = Generator(
                cuda_device_idx=int(self._cuda_idx.value),
                stage1_config=stage1_config,
                stage2_config=stage2_config
            )

            # Prepare song data
            song = Song()
            song.set_lyrics(self._lyrics_text.value)
            song.set_genre(" ".join(self._genre_selection.value))
            song.set_system_prompt(self._system_prompt.value)
            song.set_default_track_length(seconds_to_tokens(self._default_segment_length.value))

            # Transfer cache data
            cache.transfer_to_song(song)
            song.mute_segments(cache.muted_segments())

            # Initialize generation tracking
            output_song = None
            prev_stage_outputs = [song]
            final_outputs = []
            stage1_outputs = []
            stage2_outputs = []

            # Handle seed
            generation_randomize_seed = self._generation_randomize_seed.value
            generation_seed = self._seed.value
            generation_batches = int(self._generation_batches.value)

            def get_seed():
                return random.randint(0, 4294967294) if generation_randomize_seed else generation_seed

            generation_seeds = [get_seed() for _ in range(generation_batches)]

            # Stage 1 Generation
            if GenerationStage.Stage1 in generation_stages:
                with torch.no_grad():
                    try:
                        generator.load_stage1_first()
                        if params.use_audio_prompt or params.use_dual_tracks_prompt:
                            song.set_audio_prompt(generator.get_stage1_audio_prompt(params=params))
                        
                        generator.load_stage1_second()
                        
                        for ibatch in range(generation_batches):
                            if not token():
                                raise Exception("Generation stopped")
                                
                            generator.set_seed(generation_seeds[ibatch])
                            output_song = generator.generate_stage1(input=song, params=params)
                            stage1_outputs.append(output_song)
                            
                    except Exception as e:
                        token.stop_generation()
                        raise gr.Error(f"Stage 1 generation failed: {str(e)}")
                    finally:
                        generator.unload_stage1()
                        
                prev_stage_outputs = stage1_outputs

            # Stage 2 Generation
            if GenerationStage.Stage2 in generation_stages:
                with torch.no_grad():
                    try:
                        generator.load_stage2()
                        
                        for ibatch, stage1_output in enumerate(prev_stage_outputs):
                            if not token():
                                raise Exception("Generation stopped")
                                
                            generator.set_seed(generation_seeds[ibatch])
                            stage1_output.restore_muted_segments()
                            output_song = generator.generate_stage2(input=stage1_output, params=params)
                            stage2_outputs.append(output_song)
                            
                    except Exception as e:
                        token.stop_generation()
                        raise gr.Error(f"Stage 2 generation failed: {str(e)}")
                    finally:
                        generator.unload_stage2()
                        
                prev_stage_outputs = stage2_outputs

            # Post-processing
            with torch.no_grad():
                try:
                    generator.load_post_process()
                    
                    for ibatch, post_process_input in enumerate(prev_stage_outputs):
                        if not token():
                            raise Exception("Generation stopped")
                            
                        post_process_input.restore_muted_segments()
                        files = generator.post_process(
                            input=post_process_input,
                            stage_idx=output_stage.value[0],
                            output_name=f"output_{ibatch}",
                            params=params
                        )
                        
                        # Save generation state
                        generation_cache = GenerationCache.create_from_song(post_process_input)
                        generation_cache.set_muted_segments(cache.muted_segments())
                        generation_state = {
                            "cache": generation_cache.save(),
                            "generation_seed": generation_seeds[ibatch]
                        }
                        
                        final_outputs.append((files[0], generation_cache, generation_state))
                        
                except Exception as e:
                    token.stop_generation()
                    raise gr.Error(f"Post-processing failed: {str(e)}")
                finally:
                    generator.unload_post_process()

            token.stop_generation()
            return gr.update(value="Generation complete"), final_outputs

        except Exception as e:
            if token:
                token.stop_generation()
            return gr.update(value=f"Error: {str(e)}"), None

    def _on_timeline_select(self, timeline):
        """Handle timeline item selection"""
        return timeline.get("items", [])

    def _update_timeline(self, cache: GenerationCache):
        """Update timeline with current cache data"""
        timeline_items = []

        # Add cache tracks
        for istage in range(Song.NrStages):
            track_length = len(cache.track(istage, 0))
            if track_length == 0:
                continue

            elem_size = 8 if istage == 1 else 1
            timeline_items.append({
                "content": f"Stage {istage + 1}",
                "group": self.CacheTimelineGroup,
                "subgroup": f"Stage {istage + 1}",
                "start": "0",
                "end": str(tokens_to_ms(track_length//elem_size)),
                "editable": False,
                "selectable": False,
            })

        # Add segments
        for iseg, segment in enumerate(cache.segments()):
            name, start_token, end_token = segment
            track_length = len(cache.track(0, 0))
            
            if track_length == 0 or start_token > track_length:
                continue
                
            if end_token > track_length:
                end_token = track_length

            timeline_items.append({
                "id": iseg,
                "content": name,
                "group": self.SongTimelineGroup,
                "start": str(tokens_to_ms(start_token)),
                "end": str(tokens_to_ms(end_token)),
                "className": "color-primary-900" if not cache.is_muted(iseg) else "",
            })

        return VisTimelineData.model_validate_json(json.dumps({
            "items": timeline_items,
            "groups": self._timeline_groups,
        }))

    def _update_timeline_from_event(self, timeline: VisTimeline, cache: GenerationCache):
        """Update cache based on timeline changes"""
        new_segments = []
        for item in timeline.items:
            if item.group == self.SongTimelineGroup:
                new_segments.append((
                    item.content,
                    ms_to_tokens(date_to_milliseconds(item.start)),
                    ms_to_tokens(date_to_milliseconds(item.end))
                ))

        current_segments = [(segment[0], tokens_to_ms(segment[1]), tokens_to_ms(segment[2])) 
                          for segment in cache.segments()]

        if len(current_segments) != len(new_segments):
            return cache

        # Find modified segment
        modified_idx = None
        modified_segment = None
        for iseg, (new_seg, curr_seg) in enumerate(zip(new_segments, current_segments)):
            if new_seg != curr_seg:
                modified_idx = iseg
                modified_segment = new_seg
                break

        if modified_segment is None:
            return cache

        # Apply timeline constraints
        max_segment_length = tokens_to_ms(len(cache.track(0, 0)))
        name, start, end = modified_segment

        # Constrain start/end times
        start = max(start, modified_idx * self.MinTimelineBlockMs)
        start = min(start, max_segment_length - (len(current_segments) - modified_idx) * self.MinTimelineBlockMs)
        end = max(end, start + self.MinTimelineBlockMs)
        end = min(end, max_segment_length - (len(current_segments) - modified_idx - 1) * self.MinTimelineBlockMs)

        # Update preceding segments
        for iseg in range(modified_idx - 1, -1, -1):
            curr_name, curr_start, _ = new_segments[iseg]
            next_start = new_segments[iseg + 1][1]
            
            if curr_start + self.MinTimelineBlockMs > next_start:
                curr_start = next_start - self.MinTimelineBlockMs
                
            new_segments[iseg] = (curr_name, curr_start, next_start)

        # Update succeeding segments
        for iseg in range(modified_idx + 1, len(new_segments)):
            curr_name, _, curr_end = new_segments[iseg]
            prev_end = new_segments[iseg - 1][2]
            
            if prev_end + self.MinTimelineBlockMs > curr_end:
                curr_end = prev_end + self.MinTimelineBlockMs
                
            new_segments[iseg] = (curr_name, prev_end, curr_end)

        # Update cache with adjusted segments
        cache.set_segments([(seg[0], ms_to_tokens(seg[1]), ms_to_tokens(seg[2])) 
                          for seg in new_segments])

        return cache

    def _toggle_mute_segments(self, selected_items: list, cache: GenerationCache):
        """Toggle mute state for selected segments"""
        for iseg in selected_items:
            if iseg < len(cache.segments()) - 1:
                cache.toggle_mute(iseg)
        return cache

    def _split_segment(self, lyrics_text: str, cache: GenerationCache):
        """Split the last segment based on lyrics structure"""
        segments = parse_lyrics(lyrics_text)
        if cache.segments():
            _, start, end = cache.segments()[-1]
            if tokens_to_ms(end - start) >= self.MinTimelineBlockMs * 2:
                nr_segments = len(cache.segments())
                if nr_segments < len(segments):
                    cache.split_last_segment(segments[nr_segments].name())
        return cache

    def _remove_segment(self, cache: GenerationCache):
        """Remove the last segment"""
        cache.remove_last_segment()
        return cache

    def _get_model_choices(self):
        """Get available model choices from the model directory"""
        model_dir = os.path.join(model_path, "YuE")
        if not os.path.exists(model_dir):
            return []
        
        models = []
        for name in os.listdir(model_dir):
            path = os.path.join(model_dir, name)
            if os.path.isdir(path):
                models.append(name)
        
        return models

    def _on_audio_prompt_mode_change(self, mode):
        """Handle audio prompt mode changes"""
        mode = AudioPromptMode.from_string(mode)
        if mode == AudioPromptMode.Off:
            return [gr.update(visible=False)] * 4
        elif mode == AudioPromptMode.SingleTrack:
            return [
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False)
            ]
        elif mode == AudioPromptMode.DualTrack:
            return [
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True)
            ]

    def _on_generate_start(self):
        """Handle generation start"""
        token = GenerationToken()
        token.start_generation()
        return [
            token,
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(visible=True, value="Starting generation...")
        ]

    def _on_generate_complete(self):
        """Handle generation completion"""
        return [
            None,
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(visible=False)
        ]

    def _on_generate_stop(self, token):
        """Handle generation stop"""
        if token:
            token.stop_generation(False, "Cancelled")

    def _update_timeline_options_visibility(self, selection, cache):
        """Update visibility of timeline option buttons"""
        has_selection = len(selection) > 0
        can_edit_last = len(selection) == 1 and len(cache.segments()) - 1 in selection
        
        return [
            gr.update(visible=has_selection),
            gr.update(visible=can_edit_last)
        ]

    def _create_audio_players(self, max_players):
        """Create audio players with custom wavesurfer visualization"""
        self._players = []
        for i in range(max_players):
            with gr.Column(visible=False) as player_column:
                # Custom HTML for audio player with wavesurfer
                gr.HTML(f"""
                    <div class="audio-player" id="player-{i}">
                        <div class="controls">
                            <button class="play-button" onclick="togglePlay(this)">
                                <i class="fa fa-play"></i>
                            </button>
                            <div class="time-display">
                                <span class="current-time">0:00</span>
                                <span>/</span>
                                <span class="total-time">0:00</span>
                            </div>
                        </div>
                        <div class="waveform"></div>
                        <div class="progress-indicator"></div>
                    </div>
                """)
                
                with gr.Row():
                    accept_button = gr.Button("Accept", elem_classes=["accept-button"])
                    reject_button = gr.Button("Reject", elem_classes=["reject-button"])
                    download_button = gr.Button("Download", elem_classes=["download-button"])
                
                audio_file = gr.Audio(type="filepath", visible=False)
                
                # JavaScript to initialize the player
                gr.HTML(f"""
                    <script>
                        document.addEventListener('DOMContentLoaded', function() {{
                            initializeAudioPlayer('player-{i}');
                        }});
                    </script>
                """)
            
            self._players.append(AudioPlayer(
                column=player_column,
                accept_button=accept_button,
                reject_button=reject_button,
                audio_file=audio_file
            ))

def register_descriptions(arg_handler: ArgHandler):
    """Register descriptions for YuE UI components with the argument handler"""
    descriptions = {
        "yue_model_language": "Select the language model to use for generation",
        "yue_max_new_tokens": "Maximum number of tokens to generate",
        "yue_run_n_segments": "Number of segments to generate",
        "yue_stage2_batch_size": "Batch size for Stage 2 processing",
        "yue_keep_intermediate": "Keep intermediate files during generation",
        "yue_disable_offload_model": "Disable model offloading to save memory",
        "yue_rescale": "Rescale the output audio",
        "yue_cuda_idx": "CUDA device index for GPU processing",
        "yue_seed": "Random seed for generation (-1 for random)",
        "yue_stage1_model": "Model checkpoint for Stage 1 generation",
        "yue_stage1_cache_mode": "Cache mode for Stage 1 model",
        "yue_stage2_model": "Model checkpoint for Stage 2 generation",
        "yue_stage2_cache_mode": "Cache mode for Stage 2 model",
        "yue_generation_stage_mode": "Configure which generation stages to run",
        "yue_generation_mode": "Choose between full or continued generation",
        "yue_generation_format": "Output audio format",
        "yue_generation_length": "Length of audio to generate in seconds",
        "yue_genre_selection": "Musical genre and style tags",
        "yue_lyrics_text": "Lyrics with [verse], [chorus] labels",
        "yue_system_prompt": "System prompt for Stage 1 generation",
        "yue_default_segment_length": "Default length for song segments",
        "yue_audio_prompt_mode": "Audio prompt configuration",
        "yue_audio_prompt_file": "Reference audio file for generation",
        "yue_vocal_track_prompt_file": "Vocal track for dual-track prompting",
        "yue_instrumental_track_prompt_file": "Instrumental track for dual-track prompting",
        "yue_audio_prompt_start_time": "Start time for audio prompt",
        "yue_audio_prompt_end_time": "End time for audio prompt"
    }
    
    for elem_id, description in descriptions.items():
        arg_handler.register_description("yue", elem_id, description) 