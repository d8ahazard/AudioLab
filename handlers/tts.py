import json
import logging
import os
import time

import torch
from TTS.api import TTS

from handlers.config import output_path, model_path

logger = logging.getLogger(__name__)


class TTSHandler:
    def __init__(self, language="en"):
        self.language = language
        # Get device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dict = TTS().list_models().models_dict
        self.tts_models = self.model_dict.get("tts_models", {})
        self.tts_languages = [key for key in self.tts_models.keys() if key != "multilingual"]
        self.selected_model = None
        self.default_model = "multilingual/xtts_v2"
        self.tts = None
        self.model_data = {}

        # Fetch metadata for the default model
        self.fetch_model_metadata("multilingual/xtts_v2")

    def fetch_model_metadata(self, model_name):
        full_model_path = "tts_models/" + model_name
        if "multilingual" in full_model_path:
            lang = "multilingual"
        else:
            lang = self.language
        try:
            self.model_data = self.tts_models.get(lang, {}).get(
                model_name.split("/")[0], {}).get(model_name.split("/")[1], {}
                                                  )
        except:
            self.model_data = {}
        logger.info(f"Fetched metadata for model: {full_model_path}, data: {self.model_data}")

    def handle(self, text: str, model_name: str, speaker_wav: str, selected_speaker: str, speed: float = 1.0):
        output_dir = os.path.join(output_path, "tts")
        # Use timestamp to make the filename unique
        file_stamp = str(int(time.time()))
        output_file = os.path.join(output_dir, f"(TTS)_{file_stamp}.wav")
        os.makedirs(output_dir, exist_ok=True)
        full_model_path = "tts_models/" + model_name
        self.load_model(model_name)
        lang = self.language if "multilingual" in full_model_path else None
        self.tts.tts_to_file(text=text, speaker_wav=speaker_wav, file_path=output_file, language=lang, speed=speed,
                             speaker=selected_speaker)
        logger.info(f"Output file: {output_file}")
        if self.device == "cuda":
            self.tts.to("cpu")
            torch.cuda.empty_cache()
        return output_file

    def available_models(self):
        language_models = self.tts_models.get(self.language, {})
        multilingual_models = self.tts_models.get("multilingual", {})
        all_model_keys = []
        for model_name, sub_models in language_models.items():
            for sub_model, model_data in sub_models.items():
                all_model_keys.append(self.language + "/" + model_name + "/" + sub_model)
        for model_name, sub_models in multilingual_models.items():
            for sub_model, model_data in sub_models.items():
                all_model_keys.append("multilingual/" + model_name + "/" + sub_model)
        return all_model_keys

    def load_model(self, model_name):
        full_model_path = "tts_models/" + model_name
        if self.selected_model != full_model_path or not self.tts:
            logger.info(f"Loading model: {full_model_path}")
            self.tts = TTS(model_name=full_model_path).to(self.device)
            self.selected_model = full_model_path
        if self.device == "cuda":
            self.tts.to("cuda")
        return self.tts

    def available_languages(self):
        return self.tts_languages

    def available_speakers(self):
        speaker_config_file = None
        if self.selected_model is not None:
            model = os.path.join(self.selected_model.split("/")[0], self.selected_model.split("/")[1])
            speaker_config_file = os.path.join(model_path, "tts", model, "speakers.json")
        if self.default_model is not None:
            model = os.path.join(self.default_model.split("/")[0], self.default_model.split("/")[1])
            speaker_config_file = os.path.join(model_path, "tts", model, "speakers.json")
        if speaker_config_file and os.path.exists(speaker_config_file):
            try:
                with open(speaker_config_file, "r") as f:
                    speakers = json.load(f)
                return speakers
            except Exception as e:
                logger.error(f"Error fetching speakers: {e}")
        else:
            logger.info(f"Loading default model for speaker fetch.")
            self.load_model(self.default_model)
        if self.tts and getattr(self.tts, "is_multi_speaker", False):
            speakers = getattr(self.tts, "speakers", None)
            if speakers:
                os.makedirs(os.path.dirname(speaker_config_file), exist_ok=True)
                with open(speaker_config_file, "w") as f:
                    f.write(json.dumps(speakers))
                logger.info(f"Saved speakers to {speaker_config_file}")
                return speakers
            else:
                try:
                    speakers = list(self.tts.synthesizer.tts_model.speaker_manager.name_to_id)
                    if len(speakers) > 0:
                        os.makedirs(os.path.dirname(speaker_config_file), exist_ok=True)
                        with open(speaker_config_file, "w") as f:
                            f.write(json.dumps(speakers))
                        logger.info(f"Saved speakers to {speaker_config_file}")
                    return speakers
                except Exception as e:
                    logger.error(f"Error fetching speakers: {e}")
        return []
