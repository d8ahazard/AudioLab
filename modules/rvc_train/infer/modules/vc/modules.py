import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from modules.rvc_train.lib.models import RVCModel
from modules.rvc_train.infer.lib.audio import load_audio
from modules.rvc_train.infer.modules.vc.pipeline import Pipeline
from modules.rvc_train.infer.modules.vc.utils import load_hubert

logger = logging.getLogger(__name__)


class VC:
    def __init__(self, config):
        self.config = config
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.tgt_sr = None
        self.hubert_model = None

    def get_vc(self, rmvpe_path, hubert_path):
        if self.hubert_model is None:
            self.hubert_model = load_hubert(
                hubert_path=str(hubert_path), device=self.config.device, is_half=self.config.is_half
            )

        # loads pipeline with rmvpe model
        self.pipeline = Pipeline(rmvpe_path, self.config)

    def load_generator(self, model_path):
        """
        Loads self.net_g
        """
        self.cpt = torch.load(model_path, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        logger.info(f"target sample rate from the config is: {self.tgt_sr}")
        self.pipeline.set_t_pad_tgt(self.tgt_sr)

        n_spk = self.cpt["weight"]["emb_g.weight"].shape[0]

        gen_class = RVCModel
        self.net_g = gen_class(*self.cpt["config"], is_half=self.config.is_half)
        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        self.net_g = self.net_g.half() if self.config.is_half else self.net_g.float()

    def vc_single(self,
                  sid: int, input_audio_path: str, f0_up_key: int, f0_file: Optional[str],
                  file_index_path: str, index_rate: float, resample_sr: int, rms_mix_rate: float,
                  protect: float, index_topk: int = 8, average_pitch: int = 0):
        if not input_audio_path:
            return "You need to upload an audio", None
        try:
            logger.info("Loading audio")
            audio = self._load_and_process_audio(input_audio_path)
            times = [0, 0, 0]
            file_index_path = Path(file_index_path) if file_index_path else None
            audio_opt, f0_statistics = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                times,
                f0_up_key,
                str(file_index_path) if file_index_path else "",
                index_rate,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                protect,
                f0_file,
                index_topk=index_topk,
                average_pitch=average_pitch
            )
            if self.tgt_sr != resample_sr and resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            logger.info(f"target sample rate: {tgt_sr}")

            index_info = f"Index:\n{file_index_path}." if file_index_path and file_index_path.exists() else "Index not used."
            return (
                {
                    "info": f"{index_info}\nTime:\nnpy: {times[0]:.2f}s, f0: {times[1]:.2f}s, infer: {times[2]:.2f}s.",
                    "low_f0": f0_statistics[0],
                    "high_f0": f0_statistics[1],
                    "mean_f0": f0_statistics[2]
                },
                (tgt_sr, audio_opt),
            )
        except Exception as e:
            logger.warning(str(e), exc_info=True)
            return str(e), (None, None)

    def _load_and_process_audio(self, input_audio_path):
        # resample audio and load in float32
        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        return audio
