import os
import re
from abc import abstractmethod
from typing import List, Dict, Any, Union, Tuple, Callable

import pydantic
import gradio as gr
from annotated_types import Ge, Le

from handlers.args import ArgHandler
from util.data_classes import ProjectFiles

if pydantic.__version__.startswith("1."):
    PYDANTIC_V2 = False
else:
    PYDANTIC_V2 = True


class TypedInput:
    def __init__(self, default: Any = ...,
                 description: str = None,
                 ge: float = None,
                 le: float = None,
                 step: float = None,
                 min_length: int = None,
                 max_length: int = None,
                 regex: str = None,
                 choices: List[Union[str, int]] = None,
                 type: type = None,
                 gradio_type: str = None,
                 render: bool = True,
                 required: bool = False,
                 on_click: Callable = None,
                 on_input: Callable = None,
                 on_change: Callable = None,
                 on_clear: Callable = None,
                 refresh: Callable = None,
                 ):
        field_kwargs = {
            "default": default,
            "description": description,
            "ge": ge,
            "le": le,
            "min_length": min_length,
            "max_length": max_length,
            "step": step,
        }

        if PYDANTIC_V2:
            field_kwargs["pattern"] = regex
            if choices:
                field_kwargs["json_schema_extra"] = {"enum": choices}
        else:
            field_kwargs["regex"] = regex
            field_kwargs["enum"] = choices

        field = pydantic.Field(**field_kwargs)
        self.type = type
        self.field = field
        self.render = render
        self.required = required
        self.on_click = on_click
        self.on_input = on_input
        self.on_change = on_change
        self.on_clear = on_clear
        self.refresh = refresh
        self.gradio_type = gradio_type if gradio_type else self.pick_gradio_type()

    def pick_gradio_type(self):
        if self.type == bool:
            return "Checkbox"
        if self.type == str:
            return "Text"
        if self.type == int or self.type == float:
            try:
                if self.field.ge is not None and self.field.le is not None:
                    return "Slider"
            except:
                pass
        if self.type == float:
            return "Number"
        if self.type == list:
            return "Textfield"
        return "Text"


class BaseWrapper:
    _instance = None
    priority = 1000
    allowed_kwargs = {}
    description = "Base Wrapper"
    default = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BaseWrapper, cls).__new__(cls)
            cls._instance.arg_handler = ArgHandler()
            class_name = cls.__name__
            cls._instance.title = ' '.join(
                word.capitalize() for word in re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).split('_'))

        return cls._instance

    def validate_args(self, **kwargs: Dict[str, Any]) -> bool:
        """
        Validate the provided arguments.
        """
        filtered_kwargs = {}
        for key, value in kwargs.items():
            if key in self.allowed_kwargs:
                filtered_kwargs[key] = value
        for arg, value in self.allowed_kwargs.items():
            if value.field.required and arg not in filtered_kwargs or not filtered_kwargs[arg]:
                return False
        return True

    @abstractmethod
    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        pass

    @abstractmethod
    def register_api_endpoint(self, api) -> Any:
        pass

    def render_options(self, container: gr.Column):
        """
        Render the options for this wrapper into the provided container.
        """
        if not len(self.allowed_kwargs):
            return
        for key, value in self.allowed_kwargs.items():
            if value.render:
                with container:
                    self.create_gradio_element(self.__class__.__name__, key, value)

    def create_gradio_element(self, class_name: str, key: str, value: TypedInput):
        """
        Create and register a Gradio element for the specified input field.
        """
        arg_key = key
        key = " ".join([word.capitalize() for word in key.split("_")])

        # Extract `ge` and `le` from metadata if they exist
        ge_value = None
        le_value = None
        choices = None
        step = None
        for meta in value.field.metadata:
            if isinstance(meta, Ge):
                ge_value = meta.ge
            elif isinstance(meta, Le):
                le_value = meta.le

        if value.field.json_schema_extra:
            for extra in value.field.json_schema_extra:
                if extra == "enum":
                    choices = value.field.json_schema_extra[extra]
                if extra == "step":
                    step = value.field.json_schema_extra[extra]

        match value.gradio_type:
            case "Checkbox":
                elem = gr.Checkbox(label=key, value=value.field.default)
            case "Text":
                elem = gr.Textbox(label=key, value=value.field.default)
            case "Slider":
                elem = gr.Slider(
                    label=key,
                    value=value.field.default,
                    minimum=ge_value if ge_value is not None else 0,
                    maximum=le_value if le_value is not None else 100,
                    step=step if step is not None else 1 if isinstance(value.field.default, int) else 0.1
                )
            case "Number":
                elem = gr.Number(label=key, value=value.field.default)
            case "Textfield":
                elem = gr.Textbox(label=key, value=value.field.default, lines=3)
            case "Dropdown":
                elem = gr.Dropdown(label=key, choices=choices, value=value.field.default)
            case "File":
                elem = gr.File(label=key)
            case _:
                elem = gr.Textbox(label=key, value=value.field.default)

        self.arg_handler.register_element(class_name, arg_key, elem)
        if isinstance(value.on_click, Callable):
            getattr(elem, "click")(value.on_click)
        if isinstance(value.on_input, Callable):
            getattr(elem, "input")(value.on_input)
        if isinstance(value.on_change, Callable):
            getattr(elem, "change")(value.on_change)
        if isinstance(value.on_clear, Callable):
            getattr(elem, "clear")(value.on_clear)
        if isinstance(value.refresh, Callable):
            # Create a gradio button as well, to trigger the refresh
            refresh_button = gr.Button(value=f"Refresh {key}")
            getattr(refresh_button, "click")(fn=value.refresh, outputs=[elem])
            print(f"Registering refresh button for {key}")
            return elem, refresh_button
        return elem

    @staticmethod
    def filter_inputs(inputs: List[ProjectFiles], input_type: str = "audio") -> Tuple[List[str], List[str]]:
        """
        Filter the inputs to only include files that exist.
        """
        filtered_inputs, outputs = [], []
        extensions = []
        match input_type:
            case "audio":
                extensions = ["mp3", "wav", "flac", "m4a", "aac", "ogg", "opus"]
            case "image":
                extensions = ["jpg", "jpeg", "png"]
            case "video":
                extensions = ["mp4", "mov", "avi"]
            case "text":
                extensions = ["txt", "csv", "json"]
            case _:
                print(f"Unknown input type: {input_type}")
                pass
        if input_type == "audio":
            extensions = ["mp3", "wav", "flac", "m4a", "aac", "ogg", "opus"]
        if input_type == "image":
            extensions = ["jpg", "jpeg", "png"]
        if input_type == "video":
            extensions = ["mp4", "mov", "avi"]
        if input_type == "text":
            extensions = ["txt", "csv", "json"]
        if len(extensions) == 0:
            return filtered_inputs, outputs

        for input_file in inputs:
            file, ext = os.path.splitext(input_file)
            if ext[1:] in extensions:
                filtered_inputs.append(input_file)
            else:
                outputs.append(input_file)
        return filtered_inputs, outputs
