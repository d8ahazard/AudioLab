from collections import OrderedDict
from collections.abc import Callable
import copy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import os
import random
from tempfile import NamedTemporaryFile

from infer import GenerationToken, GenerationParams, Generator, Stage1Config, Stage2Config
from song import Song, GenerationCache, parse_lyrics

import gradio as gr
from gradio_vistimeline import VisTimeline, VisTimelineData
from tqdm import tqdm
import torch

def date_to_milliseconds(date):
    try:
        date = int(date)
    except ValueError:
        pass
    if isinstance(date, int):  # Input is already in milliseconds (Unix timestamp)
        return date
    elif isinstance(date, str):  # Input is ISO8601 datetime string
        dt = datetime.fromisoformat(date.replace("Z", "+00:00"))
        epoch = datetime(1970, 1, 1, tzinfo=dt.tzinfo) # Calculate difference from Unix epoch
        return int((dt - epoch).total_seconds() * 1000)
    else:
        return 0  # Fallback for unsupported types

def tokens_to_ms(nr_tokens: int):
    return nr_tokens * 1000 // 50

def ms_to_tokens(milliseconds: int):
    return (milliseconds * 50) // 1000

def seconds_to_tokens(seconds: int):
    return seconds * 50

def tokens_to_seconds(tokens: int):
    return tokens // 50

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

# Functions to List and Categorize Models
def get_models(model_dir):
    """
    Lists all models in the specified directory and categorizes them as Stage1, Stage2, or both.
    """
    if not os.path.isdir(model_dir):
        return [], [], []
    
    # List directories only
    models = [name for name in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, name))]
    stage1_models = []
    stage2_models = []
    both_stage_models = []

    for model in models:
        lower_name = model.lower()
        model_path = os.path.join(model_dir, model)
        if 's1' in lower_name:
            stage1_models.append(model_path)
        if 's2' in lower_name:
            stage2_models.append(model_path)
        if 's1' not in lower_name and 's2' not in lower_name:
            both_stage_models.append(model_path)
    return stage1_models, stage2_models, both_stage_models

def load_and_process_genres(json_path):
    """
    Loads JSON data, processes genres, timbres, genders, moods, and instruments,
    removes duplicates (case insensitive), and returns a sorted list of unique values.
    """
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Combine all relevant categories into a single list
    categories = ['genre', 'timbre', 'gender', 'mood', 'instrument']
    all_items = [item.strip() for category in categories for item in data.get(category, [])]
    
    # Use a set for deduplication (case insensitive)
    unique_items = OrderedDict()
    for item in all_items:
        key = item.lower()
        if key not in unique_items and item:  # Skip empty strings
            unique_items[key] = item
    
    # Sort alphabetically while preserving original capitalization
    sorted_items = sorted(unique_items.values(), key=lambda x: x.lower())
    
    return sorted_items

class AppMain:

    AllowedPaths: list = ["outputs"]
    MaxBatches: int = 10

    CacheTimelineGroup: int = 0
    SongTimelineGroup: int = 1

    MinTimelineBlockMs: int = 20

    DefaultStage1Model: str = "YuE-s1-7B-anneal-en-cot-exl2"
    DefaultStage1CacheMode: str = "Q4"

    DefaultStage2Model: str = "YuE-s2-1B-general-exl2"
    DefaultStage2CacheMode: str = "FP16"

    def __init__(self,
                 server_name: str = "127.0.0.1",
                 server_port: int = 7860,
                 working_directory: str = "",
                 ):
        
        self._server_name = server_name
        self._server_port = server_port
        self._working_directory = working_directory

        os.environ["GRADIO_TEMP_DIR"] = os.path.abspath(os.path.join(self._working_directory, "tmp"))

        self._allowed_paths = [os.path.abspath(os.path.join(self._working_directory, path)) for path in AppMain.AllowedPaths]
        
        self._output_dir = os.path.join(self._working_directory, "outputs")
        self._model_dir = os.path.join(self._working_directory, "models")

        gr.set_static_paths(paths=["icons", "scripts"])

        self._component_serializers = {}
        self._interface = self.create_interface()

    def save_state(self, *args):
        output = {}

        for i, (name, serializer) in enumerate(self._component_serializers.items()):
            output[name] = serializer.save(args[i])

        return output

    def load_state(self, saved_data: dict):
        outputs = []

        if saved_data:
            for name, serializer in self._component_serializers.items():
                if name in saved_data:
                    outputs.append(serializer.load(saved_data[name]))
                else:
                    outputs.append(gr.skip())
        else:
            for _ in enumerate(self._component_serializers):
                outputs.append(gr.skip())

        return outputs

    def save_state_file(self, saved_state: dict):

        with NamedTemporaryFile(
            prefix="song_", 
            suffix=".json", 
            dir=self._output_dir, 
            mode="w",
            encoding="utf-8",
            delete=False) as file:
            file.write(json.dumps(saved_state, indent=2))
            return file.name

        return ""

    def load_state_file(self, filename):
        with open(filename, "r") as file:
            return json.loads(file.read())
        return {}

    def serialized_components(self):
        return [serializer.component for serializer in self._component_serializers.values()]

    def read_state_value(self,
        saved_data: dict,
        component: gr.Component):
        """
        Read component value saved_data
        """
        for name, serializer in self._component_serializers.items():
            if serializer.component == component:
                assert(name in saved_data)
                return serializer.load(saved_data[name])
        return None

    def S(self,
                identifier: str,
                component: gr.Component,
                load: Callable = None,
                save: Callable = None):
        """
        Serialize component state
        """

        assert(identifier not in self._component_serializers)

        @dataclass
        class ComponentSerializer:
            component:object
            load:Callable
            save:Callable

        self._component_serializers[identifier] = ComponentSerializer(
            component = component,
            load = load if load else lambda x: x,
            save = save if save else lambda x: x,
        )

        return component

    def get_time_mmsscc(self, value):
        minutes = int(value // 60)
        seconds = int(value - minutes * 60)
        cent = int(round(100 * (value - minutes * 60 - seconds)))
        return f"{minutes:02}:{seconds:02}:{cent:02}"
    
    def song_data_cache_to_timeline(self, timeline: VisTimeline, cache: GenerationCache):

        stage_names = ("Stage 1", "Stage 2")
        timeline_items = []

        for istage in range(Song.NrStages):
            
            track_length = len(cache.track(istage, 0))
            if track_length == 0:
                continue

            elem_size = 8 if istage == 1 else 1

            timeline_items.append({
                "content" : stage_names[istage],
                "group" : AppMain.CacheTimelineGroup,
                "subgroup": stage_names[istage],
                "start" : "0",
                "end" : str(tokens_to_ms(track_length//elem_size)),
                "editable" : False,
                "selectable": False,
            })

        for iseg, segment in enumerate(cache.segments()):
            name, start_token, end_token = segment
            start_token = start_token
            end_token = end_token
            track_length = len(cache.track(0, 0))
            if track_length == 0 or start_token > track_length:
                continue
            if end_token > track_length:
                end_token = track_length

            timeline_items.append({
                "id" : iseg,
                "content" : name,
                "group" : AppMain.SongTimelineGroup,
                "start" : str(tokens_to_ms(start_token)),
                "end" : str(tokens_to_ms(end_token)),
                "className": "color-primary-900" if not cache.is_muted(iseg) else "",
            })

        value = {
            "items" : timeline_items,
            "groups": self._timeline_groups,
        }

        return VisTimelineData.model_validate_json(json.dumps(value))

    def timeline_to_song_data_cache(self, timeline: VisTimeline, cache: GenerationCache):

        new_segments = []
        for item in timeline.items:
            if item.group == AppMain.SongTimelineGroup:
                new_segments.append((item.content, date_to_milliseconds(item.start), date_to_milliseconds(item.end)))

        segments = [(segment[0], tokens_to_ms(segment[1]), tokens_to_ms(segment[2])) for segment in cache.segments()]

        assert(len(segments) == len(new_segments))

        max_segments = len(segments)

        def modified_segment(segments):
            for iseg, seg in enumerate(new_segments):
                if seg != segments[iseg]:
                    return iseg, seg
            return None, None

        modified_segment_idx, modified_segment = modified_segment(segments)

        if not modified_segment:
            return cache

        max_segment_length = tokens_to_ms(len(cache.track(0, 0)))

        mod_segment_name, mod_segment_start, mod_segment_end = new_segments[modified_segment_idx]

        mod_segment_min_start_pos = modified_segment_idx * AppMain.MinTimelineBlockMs if modified_segment_idx > 0 else 0
        mod_segment_max_end_pos = max_segment_length - (max_segments - modified_segment_idx - 1) * AppMain.MinTimelineBlockMs

        mod_segment_start = max(mod_segment_start, mod_segment_min_start_pos)
        mod_segment_start = min(mod_segment_start, mod_segment_max_end_pos - AppMain.MinTimelineBlockMs)
        mod_segment_end = max(mod_segment_end, mod_segment_start + AppMain.MinTimelineBlockMs)
        mod_segment_end = min(mod_segment_end, mod_segment_max_end_pos)

        new_segments[modified_segment_idx] = (mod_segment_name, mod_segment_start, mod_segment_end)

        # Adjust preceding segments
        for iseg in range(modified_segment_idx - 1, -1, -1):
            _, succeeding_seg_start, _ = new_segments[iseg + 1]
            name, start, end = new_segments[iseg]

            # Preceding block end must be aligned with succeeding block start
            end = succeeding_seg_start
            # Ensure block is at least AppMain.MinTimelineBlockMs
            if start + AppMain.MinTimelineBlockMs > end:
                start = end - AppMain.MinTimelineBlockMs

            new_segments[iseg] = (name, start, end)
        
        # Adjust succeeding segments
        for iseg in range(modified_segment_idx + 1, max_segments):
            _, _, preceeding_seg_end = new_segments[iseg - 1]
            name, start, end = new_segments[iseg]

            # Succeeding block start must be aligned with precceding block end
            start = preceeding_seg_end
            # Ensure block is at least AppMain.MinTimelineBlockMs
            if start + AppMain.MinTimelineBlockMs > end:
                end = start + AppMain.MinTimelineBlockMs

            new_segments[iseg] = (name, start, end)

        adjusted_segments = [(segment[0], ms_to_tokens(segment[1]), ms_to_tokens(segment[2])) for segment in new_segments]
        cache.set_segments(adjusted_segments)

        return cache

    def timeline_select_item(self, timeline, event_data: gr.EventData):
        return event_data._data

    def cache_split_segment(self, lyrics_text: str, cache: GenerationCache):
        segments = parse_lyrics(lyrics_text)
        if cache.segments():
            _, start, end = cache.segments()[-1]
            if tokens_to_ms(end - start) >= AppMain.MinTimelineBlockMs * 2:
                nr_segments = len(cache.segments())
                if nr_segments < len(segments):
                    cache.split_last_segment(segments[nr_segments].name())

    def cache_remove_segment(self, cache: GenerationCache):
        cache.remove_last_segment()

    def toggle_mute_selected_timeline_items(self, muted_items: list, cache: GenerationCache):
        for iseg in muted_items:
            if iseg < len(cache.segments()) - 1:
                cache.toggle_mute(iseg)
        return cache

    def create_interface(self):
        theme = gr.themes.Base()

        css = ""
        with open("scripts/style.css") as file:
            css = file.read()

        head="""
        <script type="module" src="/gradio_api/file=scripts/wavesurfer.esm.js"></script>
        <script type="module" src="/gradio_api/file=scripts/audioplayer.js"></script>
        <script type="module" src="/gradio_api/file=scripts/utils.js"></script>
        """

        with gr.Blocks(head=head, css=css, title="YuE - UI", theme=theme) as interface:
            gr.Markdown("# YuE - UI")

            self.create_states()
            
            with gr.Row():
                with gr.Column():
                    with gr.Tab("Prompt"):
                        self.create_prompt_tab()
                    with gr.Tab("Model"):
                        self.create_model_tab()

                with gr.Column():
                    with gr.Tab("Generation"):
                        self.create_generation_tab()

            self.create_sidebar()

        return interface

    def create_states(self):
        self._generation_token = gr.State()
        self._generation_outputs = gr.State()
        self._selected_timeline_items = gr.State([])

        def load_cache(cache_data):
            cache = GenerationCache(Song.NrStages)
            if cache_data:
                cache.load(cache_data)
            return cache

        def save_cache(cache):
            return cache.save()

        self._generation_cache = self.S("cache", gr.State(GenerationCache()), load=load_cache, save=save_cache)

    def create_prompt_tab(self):

        with gr.Accordion(label="Text prompt"):

            genres = load_and_process_genres(os.path.join(self._working_directory, "top_200_tags.json"))
            self._genre_selection = self.S("genre_text", gr.Dropdown(
                label="Select Music Genres",
                info="Select genre tags that describe the musical style or characteristics (e.g., instrumental, genre, mood, vocal timbre, vocal gender). This is used as part of the generation prompt.",
                choices=genres,
                interactive=True,
                multiselect=True,
                allow_custom_value=True,
                max_choices=50
            ))

            self._lyrics_text = self.S("lyrics_text", gr.Textbox(
                label="Lyrics Text",
                lines=4,
                placeholder="Type the lyrics here...",
                info="Text containing the lyrics for the music generation. These lyrics will be processed and split into structured segments to guide the generation process.",
                value=""
            ))

            self._default_segment_length = self.S("default_segment_length", gr.Number(
                label="Segment length",
                value=30,
                info="Preferred song segment length in seconds. The generated segment length might be shorter."
            ))

            self._system_prompt = self.S("system_prompt", gr.Textbox(
                label="Stage 1 system prompt",
                value="Generate music from the given lyrics segment by segment.",
                info="The system prompt to be used for stage 1 generation.",
            ))

        with gr.Accordion("Audio prompt"):

            self._audio_prompt_mode = self.S("audio_prompt", gr.Radio(
                label="Audio prompt mode",
                choices=[str(mode) for mode in AudioPromptMode],
                value=str(AudioPromptMode.Off),
                info="Select whether to use a single audio track or instrumental + vocal tracks."
                ))

            with gr.Column(visible=False) as audio_prompt:

                self._audio_prompt_file = self.S("audio_prompt_file", gr.File(
                    label="Upload Audio File",
                    file_types=["audio"],
                    file_count="single",
                ))
                
                self._vocal_track_prompt_file = self.S("vocal_track_prompt_file", gr.File(
                    label="Upload Vocal Track File",
                    file_types=["audio"],
                    file_count="single",
                    visible=False,
                ))
                
                self._instrumental_track_prompt_file = self.S("instrumnet_track_prompt_file", gr.File(
                    label="Upload Instrumental Track File",
                    file_types=["audio"],
                    file_count="single",
                    visible=False,
                ))

                self._audio_prompt_start_time = self.S("audio_prompt_start_time", gr.Number(
                    label="Prompt Start Time (s)",
                    value=0,
                    info="The start time in seconds to extract the audio prompt from the given audio file."
                ))

                self._audio_prompt_end_time = self.S("audio_prompt_end_time", gr.Number(
                    label="Prompt End Time (s)",
                    value=30,
                    info="The end time in seconds to extract the audio prompt from the given audio file."
                ))
                
                def on_audio_prompt_mode_change(mode):
                    mode = AudioPromptMode.from_string(mode)
                    if mode == AudioPromptMode.Off:
                        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                    elif mode == AudioPromptMode.SingleTrack:
                        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
                    elif mode == AudioPromptMode.DualTrack:
                        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)                    

                self._audio_prompt_mode.change(
                    fn=on_audio_prompt_mode_change,
                    inputs=self._audio_prompt_mode,
                    outputs=[audio_prompt, self._audio_prompt_file, self._vocal_track_prompt_file, self._instrumental_track_prompt_file]
                )

    def create_model_tab(self):

        stage1_choices, stage2_choices, both_choices = get_models(self._model_dir)
        stage1_choices = stage1_choices + both_choices
        stage2_choices = stage2_choices + both_choices

        default_stage1_model = os.path.join(self._model_dir, AppMain.DefaultStage1Model)
        default_stage2_model = os.path.join(self._model_dir, AppMain.DefaultStage2Model)

        with gr.Accordion(label="Model"):
            self._stage1_model = self.S("stage1_model", gr.Dropdown(
                label="Stage1 Model",
                choices=stage1_choices,
                value=default_stage1_model,
                info="Select the checkpoint path for the Stage 1 model.",
                interactive=True
            ))

            self._stage1_model_cache_mode = self.S("stage1_model_cache_mode", gr.Dropdown(
                choices=["FP16", "Q8", "Q6", "Q4"],
                label="Select the cache quantization of the Stage1 model",
                value=AppMain.DefaultStage1CacheMode,
                interactive=True,
            ))

            self._stage1_model_cache_size = self.S("stage1_model_cache_size", gr.Number(
                label="Stage 1 Cache Size",
                value=3500,
                precision=0,
                info="The cache size used in Stage 1. This is the max context length, should use as large value as vram permits"
            ))

            self._stage2_model = self.S("stage2_model", gr.Dropdown(
                label="Stage2 Model",
                choices=stage2_choices,
                value=default_stage2_model,
                info="Select the checkpoint path for the Stage 2 model.",
                interactive=True
            ))

            self._stage2_model_cache_mode = self.S("stage2_model_cache_mode", gr.Dropdown(
                choices=["FP16", "Q8", "Q6", "Q4"],
                label="Select the cache quantization of the Stage2 model",
                value=AppMain.DefaultStage2CacheMode,
                interactive=True,
            ))

            self._stage2_model_cache_size = self.S("stage2_model_cache_size", gr.Number(
                label="Stage 2 Cache Size",
                value=6000,
                precision=0,
                info="The cache size used in Stage 2"
            ))

            self._cuda_idx = self.S("cuda_idx", gr.Number(
                label="CUDA Index",
                value=0,
                precision=0
            ))

    def create_timeline(self):
        self._timeline_groups = [
            {"id": 0, "content": "Cache"},
            {"id": 1, "content": "Segments"},
        ]

        self._timeline = VisTimeline(
            label="Generated song",
            value={
                "groups": self._timeline_groups,
            },
            options={
                "moment": "+00:00",
                "showCurrentTime": False,
                "editable": {
                    "add": False,
                    "remove": False,
                    "updateGroup": False,
                    "updateTime": True,
                    "overrideItems" : False,
                },
                "itemsAlwaysDraggable": {
                    "item": True,
                    "range": True
                },
                "multiselect" : True,
                "showMajorLabels": False,
                "format": {
                    "minorLabels": {
                        "millisecond": "mm:ss.SSS",
                        "second": "mm:ss",
                        "minute": "mm:ss",
                        "hour": "HH:mm:ss"
                    }
                },
                "stack" : False,
                "start": 0,
                "end": 200000,
                "min": 0,
                "max": 600000,
                "snap" : None,
                "zoomMin": 1000, 
            },
            preserve_old_content_on_value_change=True
        )

        with gr.Row(visible=False) as options_row:
            self._timeline_toggle_mute = gr.Button("Toggle mute segment(s)")
            with gr.Row(visible=True) as extra_options_row:
                self._timeline_split_segment = gr.Button("Split segment")
                self._timeline_remove_segment = gr.Button("Remove segment")
                self._selected_timeline_items_extra_options_row = extra_options_row
            self._selected_timeline_items_options_row = options_row

        self._timeline_split_segment.click(
            fn=self.cache_split_segment,
            inputs=[self._lyrics_text, self._generation_cache],
        ).then(
            fn=lambda: [],
            outputs=[self._selected_timeline_items]
        ).then(
            fn=self.song_data_cache_to_timeline,
            inputs=[self._timeline, self._generation_cache],
            outputs=[self._timeline]
        )

        self._timeline_remove_segment.click(
            fn=lambda cache: cache.remove_last_segment(),
            inputs=[self._generation_cache],
        ).then(
            fn=lambda: [],
            outputs=[self._selected_timeline_items]
        ).then(
            fn=self.song_data_cache_to_timeline,
            inputs=[self._timeline, self._generation_cache],
            outputs=[self._timeline]
        )

        self._timeline_toggle_mute.click(
            fn=self.toggle_mute_selected_timeline_items,
            inputs=[self._selected_timeline_items, self._generation_cache],
            outputs=[self._generation_cache]
        ).then(
            fn=self.song_data_cache_to_timeline,
            inputs=[self._timeline, self._generation_cache],
            outputs=[self._timeline]
        )

        self._generation_cache.change(
            fn=self.song_data_cache_to_timeline,
            inputs=[self._timeline, self._generation_cache],
            outputs=[self._timeline]
        )

        self._timeline.input(
            fn=self.timeline_to_song_data_cache,
            inputs=[self._timeline, self._generation_cache],
            outputs=[self._generation_cache],
        ).then(
            fn=self.song_data_cache_to_timeline,
            inputs=[self._timeline, self._generation_cache],
            outputs=[self._timeline]
        )

        self._timeline.item_select(
            fn=self.timeline_select_item,
            inputs=[self._timeline],
            outputs=[self._selected_timeline_items]
        )
        
        self._selected_timeline_items.change(
            fn=lambda selection, cache: [
                gr.update(visible=len(selection)>0), 
                gr.update(visible=len(selection)==1 and len(cache.segments())-1 in selection)
                ],
            inputs=[self._selected_timeline_items, self._generation_cache],
            outputs=[self._selected_timeline_items_options_row, self._selected_timeline_items_extra_options_row]
        )

    def create_sidebar(self):
        with gr.Sidebar(position="right"):
            gr.Markdown("# Settings")
            self._download_button = gr.DownloadButton("💾")
            self._upload_button = gr.UploadButton("📂", type="filepath", file_count="single", file_types=[".json"])

            output_project_file = gr.File(visible=False)
            output_file_path = gr.Textbox(visible=False)
            serialized_state = gr.State()

            self._download_button.click(
                fn=self.save_state,
                inputs=self.serialized_components(),
                outputs=[serialized_state]
            ).then(
                fn=self.save_state_file,
                inputs=[serialized_state],
                outputs=[output_project_file]
            ).then(
                fn=lambda x: x.replace('\\', '/'),
                inputs=[output_project_file],
                outputs=[output_file_path],
            ).success(
                fn=None,
                js="(src)=>autoDownloadData(src)",
                inputs=[output_file_path]
            )

            self._upload_button.upload(
                fn=self.load_state_file,
                inputs=[self._upload_button],
                outputs=[serialized_state]
            ).success(
                fn = self.load_state,
                inputs=[serialized_state],
                outputs=self.serialized_components()
            )

    def create_generation_tab(self):

        with gr.Row():

            self._generation_stage_mode = self.S("generation_stage_mode", gr.Radio(
                label="Stage configuration",
                choices=[str(mode) for mode in GenerationStageMode],
                value=str(GenerationStageMode.Stage1And2),
                info="Select stage configuration. You can generate the stages independently, together or use the cached data."
            ))

            self._generation_output_format = self.S("generation_output_format", gr.Radio(
                label="Format",
                choices=[str(stage) for stage in GenerationFormat],
                value=str(GenerationFormat.Mp3),
                info="Select audio format"
            ))

        self.create_timeline()

        with gr.Row():
            self._rewind_1s = gr.Button("Rewind 1 s")
            self._rewind_5s = gr.Button("Rewind 5 s")
            self._delete_generation_cache = gr.Button("🗑️")

            def rewind(time_ms):
                def inner(generation_cache: GenerationCache):
                    new_cache = copy.deepcopy(generation_cache)
                    new_cache.rewind(time_ms)
                    return new_cache
                return inner

            self._rewind_1s.click(
                fn=rewind(1000),
                inputs=[self._generation_cache],
                outputs=[self._generation_cache]
            )

            self._rewind_5s.click(
                fn=rewind(5000),
                inputs=[self._generation_cache],
                outputs=[self._generation_cache]
            )

            self._delete_generation_cache.click(
                fn=lambda: GenerationCache(Song.NrStages),
                outputs=[self._generation_cache],
            )

        with gr.Tab("General settings"):
            self._generation_mode = self.S("generation_mode", gr.Radio(
                label="Generation mode",
                choices=[str(mode) for mode in GenerationMode],
                value=str(GenerationMode.Continue),
                info="Full will generate the whole song while continue will generate a selected portion."
            ))

            self._generation_length = self.S("generation_length", gr.Slider(
                label="Generation length",
                value=5,
                minimum=1,
                maximum=60,
                step=1,
                visible=True,
                info="Select the length of the audio to be generated in seconds.",
            ))

            self._generation_randomize_seed = self.S("generation_randomize_seed", gr.Checkbox(
                label="Randomize Seed",
                value=True,
                info="Randomize the seed for each batch.",
            ))

            self._generation_seed = self.S("generation_seed", gr.Number(
                label="Seed",
                value=42,
                precision=0,
                info="Seed for random number generation."
            ))

            # Show/hide length based on mode
            def update_mode_settings(mode):
                if mode == GenerationMode.Continue.name:
                    return gr.Slider(visible=True)
                else:
                    return gr.Slider(visible=False)

            self._generation_mode.select(fn=update_mode_settings, inputs=[self._generation_mode], outputs=[self._generation_length])

            self._generation_batches = self.S("generation_batches", gr.Slider(
                label="Batches",
                value=1,
                minimum=1,
                maximum=AppMain.MaxBatches,
                step=1,
                info="Select the number of batches to generate."
            ))

        with gr.Tab("Stage 1 settings"):
            self._generation_stage1_cfg_scale = self.S("generation_stage1_cfg_scale", gr.Slider(
                label="CFG",
                value=1.5,
                minimum=0.05,
                maximum=2.5,
                step=0.05,
                info="Select the guidance scale for classifier free guidance (CFG). A higher value adheres to the prompt more closely but suffers in quality."
            ))

            self._generation_stage1_top_p = self.S("generation_stage1_top_p", gr.Slider(
                label="top_p",
                value=0.93,
                minimum=0,
                maximum=1,
                step=0.01,
                info="A value < 1 will only keep the smallest set of the most probable tokens during generation."
            ))

            self._generation_stage1_temperature = self.S("generation_stage1_temperature", gr.Slider(
                label="Temperature",
                value=1,
                minimum=0.01,
                maximum=5,
                step=0.01,
                info="A greater value adds more randomness to the output, while a smaller value will result in more predictable output."
            ))

            self._generation_repetition_penalty = self.S("generation_repetition_penalty", gr.Slider(
                label="Repetition penalty",
                value=1.1,
                minimum=1,
                maximum=1.5,
                step=0.01,
                info="A value > 1 will reduce the likelyhood of repeated tokens."
            ))

        with gr.Row():
            self._generation_start = gr.Button("Submit")
            self._generation_stop = gr.Button("Stop", interactive=True)

        self._generation_progress = gr.Label(label="Generating", show_label=True, visible=False)

        self.make_audio_players(AppMain.MaxBatches)

        self._generation_input = gr.State({})

        run_generation_event = self._generation_start.click(
            # Save current state
            fn=self.save_state,
            inputs=self.serialized_components(),
            outputs=self._generation_input
        ).then(
            # Hide players
            fn=self.hide_players,
            inputs=None,
            outputs=[player.column for player in self._players],
        ).then(
            # Init token & set button states
            fn=lambda: (GenerationToken(), gr.Button(interactive=False), gr.Button(interactive=True), gr.Label(visible=True)),
            inputs = None,
            outputs = [self._generation_token, self._generation_start,self._generation_stop, self._generation_progress]
        ).then(
            # Generate
            fn=self.run_generate,
            inputs=[
                self._generation_token,
                self._generation_input,
            ],
            outputs=[self._generation_progress, self._generation_outputs]
        ).then(
            # Token & button states
            fn=lambda: (None, gr.Button(interactive=True), gr.Button(interactive=False), gr.Label(visible=False)),
            outputs=[self._generation_token, self._generation_start,self._generation_stop, self._generation_progress]
        )

        self._generation_outputs.change(
            # Show players
            fn=self.update_players,
            inputs=[self._generation_outputs],
            outputs=[elem for player in self._players for elem in (player.column, player.accept_button, player.reject_button, player.audio_file)],
        )

        self._generation_stop.click(
            fn=self.stop_generate,
            inputs=[self._generation_token],
            cancels=[run_generation_event]
        ).then(
            outputs = [self._generation_token]
        )

    @dataclass
    class AudioPlayer:
        column : gr.Column
        accept_button : gr.Button
        reject_button : gr.Button
        audio_file : gr.File

    def make_audio_players(self, 
                     nr_players: int):
        
        self._players = []
        accept_buttons = []
        download_buttons = []

        for i in range(nr_players):
            with gr.Column(visible=False) as player_col:
                audio_file = gr.File(label=f"Batch{i}", visible=False, interactive=False)

                def load_audio_data(player_index):
                    return """
                    function(file){
                        var event_target = document.querySelector('#wavesurfer_player%d')
                        const event = new CustomEvent("load", { detail : {"url" : file.url } })
                        event_target.dispatchEvent(event)
                    }
                    """ % (player_index)

                audio_file.change(
                    fn=None,
                    inputs=audio_file,
                    js=load_audio_data(i)
                )

                gr.HTML("""
                        <div>
                            <div class="time" id="time">0:00</div>
                            <div class="duration" id="duration">0:00</div>

                            <div class="play_button" id="play_button">
                                <div class="button_container">
                                    <div id="play">
                                        <img src="gradio_api/file=icons/Play.svg" class="button_bottom"></img>
                                        <img src="gradio_api/file=icons/Play.svg" class="button_top"></img>
                                    </div>
                                    <div id="pause" style="visibility: hidden;">
                                        <img src="gradio_api/file=icons/Pause.svg" class="button_bottom"></img>
                                        <img src="gradio_api/file=icons/Pause.svg" class="button_top"></img>
                                    </div>
                                </div>
                            </div>

                            <div class="download_button">
                                <div class="download_button_container" id="download_button">
                                    <img src="gradio_api/file=icons/Download.svg" class="button_bottom"></img>
                                    <img src="gradio_api/file=icons/Download.svg" class="button_top"></img>
                                </div>
                            </div>
                        </div>""", elem_id=f"wavesurfer_player{i}")

                with gr.Row():
                    accept_button = gr.Button(value="✔", min_width=0, size="sm")
                    accept_buttons.append(accept_button)

                    download_button = gr.DownloadButton("💾", min_width=0, size="sm")
                    download_buttons.append(download_button)

                    reject_button = gr.Button(value="❌", min_width=0, size="sm")

                    # Hide this player
                    reject_button.click(fn=lambda: gr.update(visible=False),
                                        outputs=[player_col])

                    self._players.append(self.AudioPlayer(column=player_col, accept_button=accept_button, reject_button=reject_button, audio_file=audio_file))

        # Stores the serialized project data
        output_file_state = gr.State()
        # Project data file
        output_project_file = gr.File(visible=False)
        # Path to project data file for download script
        output_file_path = gr.Textbox(visible=False)
        
        for i in range(nr_players):

            hidden_components = []

            for j in range(nr_players):
                if i == j:
                    continue

                # Hide column
                hidden_components.append(self._players[j].column)

            # Hide accept/reject buttons
            hidden_components.append(self._players[i].accept_button)
            hidden_components.append(self._players[i].reject_button)

            accept_buttons[i].click(
                # Hide other players
                fn = lambda: [gr.update(visible=False) for _,_ in enumerate(hidden_components)],
                outputs = hidden_components
            ).then(
                fn=lambda outputs, i=i: [outputs[i][1], outputs[i][2]["generation_seed"]],
                inputs=[self._generation_outputs],
                outputs=[self._generation_cache, self._generation_seed]
            )

            download_buttons[i].click(
                fn = lambda outputs, i=i: outputs[i][2],
                inputs = [self._generation_outputs],
                outputs = [output_file_state]
            ).then(
                fn = self.save_state_file,
                inputs=[output_file_state],
                outputs=[output_project_file]
            ).then(
                fn=lambda x: x.replace('\\', '/'),
                inputs=[output_project_file],
                outputs=[output_file_path],
            ).success(
                fn=None,
                js="(src)=>autoDownloadData(src)",
                inputs=[output_file_path]
            )            

    def hide_players(self):
        return [gr.update(visible=False) for _,_ in enumerate(self._players)]

    def update_players(self, generated_final_outputs):
        outputs = []
        for iplayer,_ in enumerate(self._players):
            visible = generated_final_outputs and iplayer < len(generated_final_outputs)
            outputs.append(gr.update(visible=visible))
            outputs.append(gr.update(visible=visible))
            outputs.append(gr.update(visible=visible))
            outputs.append(gr.File(visible=False, value=generated_final_outputs[iplayer][0]) if visible else gr.skip())

        return outputs

    def run_generate(self,
                     token: GenerationToken,
                     input_state: dict,
                     progress=gr.Progress(track_tqdm=True)
    ):
        def R(component):
            return self.read_state_value(saved_data=input_state, component=component)

        prompt_mode = AudioPromptMode.from_string(R(self._audio_prompt_mode))
        generation_mode = GenerationMode.from_string(R(self._generation_mode))
                
        generation_stage_mode = GenerationStageMode.from_string(R(self._generation_stage_mode))

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

        output_format = GenerationFormat.from_string(R(self._generation_output_format))

        use_audio_prompt = prompt_mode == AudioPromptMode.SingleTrack
        use_dual_tracks_prompt = prompt_mode == AudioPromptMode.DualTrack

        params = GenerationParams(
            token = token,
            max_new_tokens = seconds_to_tokens(R(self._generation_length)) if generation_mode == GenerationMode.Continue else None,
            resume = generation_mode == GenerationMode.Continue,
            use_audio_prompt = use_audio_prompt,
            use_dual_tracks_prompt = use_dual_tracks_prompt,
            prompt_start_time = R(self._audio_prompt_start_time),
            prompt_end_time = R(self._audio_prompt_end_time),
            audio_prompt_path = R(self._audio_prompt_file),
            instrumental_track_prompt_path = R(self._instrumental_track_prompt_file),
            vocal_track_prompt_path = R(self._vocal_track_prompt_file),
            stage1_guidance_scale=R(self._generation_stage1_cfg_scale),
            stage1_top_p=R(self._generation_stage1_top_p),
            stage1_temperature = R(self._generation_stage1_temperature),
            stage1_repetition_penalty = R(self._generation_repetition_penalty),
            rescale = False,
            hq_audio = output_format==GenerationFormat.Wav,
            output_dir = self._output_dir,
        )

        token.start_generation()

        stage1_config = Stage1Config(
            model_path=R(self._stage1_model),
            cache_mode=R(self._stage1_model_cache_mode),
            cache_size=R(self._stage1_model_cache_size),
        )

        stage2_config = Stage2Config(
            model_path=R(self._stage2_model),
            cache_mode=R(self._stage2_model_cache_mode),
            cache_size=R(self._stage2_model_cache_size),
        )

        progress(0, desc="Starting")

        generator = Generator(cuda_device_idx=R(self._cuda_idx),
                              stage1_config=stage1_config,
                              stage2_config=stage2_config)

        song = Song()
        song.set_lyrics(R(self._lyrics_text))

        genre_text = " ".join(R(self._genre_selection))
        song.set_genre(genre_text)
        song.set_system_prompt(R(self._system_prompt))
        song.set_default_track_length(int(seconds_to_tokens(R(self._default_segment_length))))

        cache = R(self._generation_cache)
        cache.transfer_to_song(song)
        song.mute_segments(cache.muted_segments())

        output_song = None
        prev_stage_outputs = [song]
        final_outputs = []
        stage1_outputs = []
        stage2_outputs = []

        generation_randomize_seed = R(self._generation_randomize_seed)
        generation_seed =  R(self._generation_seed)
        generation_batches = R(self._generation_batches)

        def seed():
            return int(random.randrange(4294967294)) if generation_randomize_seed else generation_seed
        
        generation_seeds = [seed() for _ in range(generation_batches)]

        if GenerationStage.Stage1 in generation_stages:
            with torch.no_grad():
                try:
                    # Only load the required dependencies to generate the audio prompt to avoiding vram spikes
                    generator.load_stage1_first()

                    song.set_audio_prompt(generator.get_stage1_audio_prompt(params=params) if params.use_audio_prompt or params.use_dual_tracks_prompt else [])

                    # Load model
                    generator.load_stage1_second()

                    for ibatch in tqdm(range(generation_batches)):

                        generator.set_seed(generation_seeds[ibatch])

                        output_song = generator.generate_stage1(
                            input=song,
                            params=params
                        )

                        if not token():
                            raise Exception("Stopped")

                        stage1_outputs.append(output_song)
                except Exception as e:
                    token.stop_generation()
                    raise gr.Error(f"Stage 1 generation failed: {str(e)}")
                finally:                
                    generator.unload_stage1()
                prev_stage_outputs = stage1_outputs

        if GenerationStage.Stage2 in generation_stages:
            with torch.no_grad():
                try:
                    generator.load_stage2()

                    for ibatch, stage1_output in tqdm(enumerate(prev_stage_outputs)):
                        
                        generator.set_seed(generation_seeds[ibatch])

                        stage1_output.restore_muted_segments()

                        output_song = generator.generate_stage2(
                            input=stage1_output,
                            params=params
                        )

                        if not token():
                            raise Exception("Stopped")

                        stage2_outputs.append(output_song)
                except Exception as e:
                    token.stop_generation()
                    raise gr.Error(f"Stage 2 generation failed: {str(e)}")
                finally:
                    generator.unload_stage2()
                prev_stage_outputs = stage2_outputs

        with torch.no_grad():
            try:
                generator.load_post_process()

                for ibatch, post_process_input in tqdm(enumerate(prev_stage_outputs)):

                    post_process_input.restore_muted_segments()

                    files = generator.post_process(
                        input=post_process_input,
                        stage_idx=output_stage.value[0],
                        output_name=f"output_{ibatch}",
                        params = params)

                    if not token():
                        raise Exception("Stopped")

                    generation_cache = GenerationCache.create_from_song(post_process_input)
                    generation_cache.set_muted_segments(cache.muted_segments())
                    generation_state = copy.deepcopy(input_state)
                    generation_state["cache"] = generation_cache.save()
                    generation_state["generation_seed"] = generation_seeds[ibatch]
                    final_outputs = final_outputs + [(files[0], generation_cache, generation_state)]
            except Exception as e:
                token.stop_generation()
                raise gr.Error(f"Post process failed: {str(e)}")
            finally:
                generator.unload_post_process()

        token.stop_generation()

        return [gr.skip(), final_outputs]

    def stop_generate(self, token):
        if token:
            token.stop_generation(False, "Cancelled")

    def launch(self):
        self._interface.launch(server_name=self._server_name, server_port=self._server_port, allowed_paths=self._allowed_paths)

if __name__ == "__main__":
    app = AppMain()
    app.launch()
