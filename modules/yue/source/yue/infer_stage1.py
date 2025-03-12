from collections.abc import Callable
import gc
import os
import random
import re

from dataclasses import dataclass
from itertools import takewhile

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from einops import rearrange
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2Sampler
from modules.yue.source.yue.mmtokenizer import _MMSentencePieceTokenizer
from modules.yue.source.yue.models.soundstream_hubert_new import SoundStream
from omegaconf import OmegaConf
from torchaudio.transforms import Resample
from tqdm import tqdm

from modules.yue.source.yue.codecmanipulator import CodecManipulator
from modules.yue.source.yue.common import get_cache_class

@dataclass
class SampleSettings:
    # Here is suggested decoding config
    top_p = 0.93
    temperature = 1
    repetition_penalty = 1.1
    guidance_scale = 1.2  # None to disable cfg

    def __init__(self, use_guidance: bool = True, repetition_penalty: float = 1.1):
        if not use_guidance:
            self.guidance_scale = None
        self.repetition_penalty = repetition_penalty

def load_audio_mono(filepath, sampling_rate: int = 16000, start_time: int = 0, end_time: int = 0):
    audio, sr = torchaudio.load(filepath)
    # Convert to mono
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)

    audio = audio.squeeze().cpu().numpy()

    start_time = start_time * sampling_rate
    end_time = end_time * sampling_rate

    if end_time == 0 or end_time > len(audio):
        end_time = len(audio)

    return torch.from_numpy(np.array([audio[start_time : end_time]]))

def encode_audio(codec_model, audio_prompt, device, target_bw=0.5):
    if len(audio_prompt.shape) < 3:
        audio_prompt.unsqueeze_(0)
    with torch.no_grad():
        raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
    raw_codes = raw_codes.transpose(0, 1)
    raw_codes = raw_codes.cpu().numpy().astype(np.int16)
    return raw_codes

class Stage1Pipeline:

    def __init__(self, device: torch.device, basic_model_config: str, resume_path: str):
        self.device = device
        self.codec_tool = CodecManipulator("xcodec", 0, 1)
        self.basic_model_config = basic_model_config
        self.resume_path = resume_path
        self.codec_model = None

        # Load tokenizer
        self.mmtokenizer = _MMSentencePieceTokenizer(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mm_tokenizer_v0.2_hf", "tokenizer.model"))
        self.start_of_segment = self.mmtokenizer.tokenize("[start_of_segment]")
        self.end_of_segment = self.mmtokenizer.tokenize("[end_of_segment]")

    def load_codec_model(self):
        if self.codec_model is not None:
            return
        model_config = OmegaConf.load(self.basic_model_config)
        assert model_config.generator.name == "SoundStream"
        self.codec_model = SoundStream(**model_config.generator.config).to(self.device)
        parameter_dict = torch.load(self.resume_path, map_location=self.device, weights_only=False)
        self.codec_model.load_state_dict(parameter_dict["codec_model"])
        self.codec_model.eval()

    def get_prompt_texts(self, genres: str, lyrics: str):
        def split_lyrics(lyrics):
            pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
            segments = re.findall(pattern, lyrics, re.DOTALL)
            structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
            return structured_lyrics

        lyrics = split_lyrics(lyrics)
        full_lyrics = "\n".join(lyrics)
        prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
        prompt_texts += lyrics
        return lyrics, prompt_texts

    def tokenize_segment_text(self,
                         segment_text: str,
                         segment_tokens: list,
                         is_first_segment: bool,
                         insert_eoa: bool):

        prefix = self.end_of_segment if not is_first_segment else []
        suffix = [self.mmtokenizer.eoa] if insert_eoa else []
        return prefix + self.start_of_segment + self.mmtokenizer.tokenize(segment_text) + [self.mmtokenizer.soa] + self.codec_tool.sep_ids + segment_tokens + suffix

    def tokenize_text(self, input_text: str):
        return self.mmtokenizer.tokenize(input_text)

    def get_audio_prompt_ids(
        self,
        use_dual_tracks_prompt: bool,
        vocal_track_prompt_path: str,
        instrumental_track_prompt_path: str,
        use_audio_prompt: bool,
        audio_prompt_path: str,
        prompt_start_time: int,
        prompt_end_time: int,
    ):
        self.load_codec_model()
        if use_dual_tracks_prompt:
            vocals_ids = load_audio_mono(vocal_track_prompt_path, start_time=prompt_start_time, end_time=prompt_end_time)
            instrumental_ids = load_audio_mono(instrumental_track_prompt_path, start_time=prompt_start_time, end_time=prompt_end_time)            
            vocals_ids = encode_audio(self.codec_model, vocals_ids, self.device, target_bw=0.5)
            instrumental_ids = encode_audio(self.codec_model, instrumental_ids, self.device, target_bw=0.5)
            vocals_ids = self.codec_tool.npy2ids(vocals_ids[0])
            instrumental_ids = self.codec_tool.npy2ids(instrumental_ids[0])
            audio_prompt_codec = rearrange([np.array(vocals_ids), np.array(instrumental_ids)], "b n -> (n b)").tolist()
            del vocals_ids
            del instrumental_ids
        elif use_audio_prompt:
            audio_prompt = load_audio_mono(audio_prompt_path, start_time=prompt_start_time, end_time=prompt_end_time)
            raw_codes = encode_audio(self.codec_model, audio_prompt, self.device, target_bw=0.5)
            audio_prompt_codec = self.codec_tool.npy2ids(raw_codes[0])
            del audio_prompt
            del raw_codes
        audio_prompt_codec_ids = [self.mmtokenizer.soa] + self.codec_tool.sep_ids + audio_prompt_codec + [self.mmtokenizer.eoa]
        sentence_ids = self.mmtokenizer.tokenize("[start_of_reference]") + audio_prompt_codec_ids + self.mmtokenizer.tokenize("[end_of_reference]")        

        del self.codec_model
        torch.cuda.empty_cache()
        gc.collect()        
        return sentence_ids

    def get_first_segment_prompt(
        self,
        segment_p: str,
        prompt_text_0: str,
        use_dual_tracks_prompt: bool,
        vocal_track_prompt_path: str,
        instrumental_track_prompt_path: str,
        use_audio_prompt: bool,
        audio_prompt_path: str,
        prompt_start_time: int,
        prompt_end_time: int,
    ):
        section_text = segment_p.replace("[start_of_segment]", "").replace("[end_of_segment]", "")
        head_id = self.mmtokenizer.tokenize(prompt_text_0)
        if use_dual_tracks_prompt or use_audio_prompt:
            head_id += self.get_audio_prompt_ids(
                use_dual_tracks_prompt,
                vocal_track_prompt_path,
                instrumental_track_prompt_path,
                use_audio_prompt,
                audio_prompt_path,
                prompt_start_time,
                prompt_end_time,
            )
        return head_id + self.start_of_segment + self.mmtokenizer.tokenize(section_text) + [self.mmtokenizer.soa] + self.codec_tool.sep_ids

    def get_segment_prompt(self, segment_p: str):
        section_text = segment_p.replace("[start_of_segment]", "").replace("[end_of_segment]", "")
        return self.end_of_segment + self.start_of_segment + self.mmtokenizer.tokenize(section_text) + [self.mmtokenizer.soa] + self.codec_tool.sep_ids

    def save(self, raw_output: torch.Tensor, output_dir: str, use_audio_prompt: bool, use_dual_tracks_prompt: bool):
        # save raw output and check sanity
        ids = raw_output[0].cpu().numpy()
        soa_idx = np.where(ids == self.mmtokenizer.soa)[0].tolist()
        eoa_idx = np.where(ids == self.mmtokenizer.eoa)[0].tolist()
        if len(soa_idx) != len(eoa_idx):
            raise ValueError(f"invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}")

        vocals = []
        instrumentals = []
        range_begin = 1 if use_audio_prompt or use_dual_tracks_prompt else 0
        for i in range(range_begin, len(soa_idx)):
            codec_ids = ids[soa_idx[i] + 1 : eoa_idx[i]]
            if codec_ids[0] == 32016:
                codec_ids = codec_ids[1:]
            codec_ids = codec_ids[: 2 * (codec_ids.shape[0] // 2)]
            vocals_ids = self.codec_tool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
            vocals.append(vocals_ids)
            instrumentals_ids = self.codec_tool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
            instrumentals.append(instrumentals_ids)
        vocals = np.concatenate(vocals, axis=1)
        instrumentals = np.concatenate(instrumentals, axis=1)
        stage1_output_dir = os.path.join(output_dir, "stage1")
        os.makedirs(stage1_output_dir, exist_ok=True)
        vocal_save_path = os.path.join(stage1_output_dir, "vtrack.npy")
        inst_save_path = os.path.join(stage1_output_dir, "itrack.npy")
        np.save(vocal_save_path, vocals)
        np.save(inst_save_path, instrumentals)

    def convert_output_to_segments(self,
             tokens: torch.Tensor,
             skip_first_block: bool):

        # save raw output and check sanity
        ids = tokens[0].cpu().numpy()

        soa_idx = np.where(ids == self.mmtokenizer.soa)[0].tolist()
        eoa_idx = np.where(ids == self.mmtokenizer.eoa)[0].tolist()
        if len(soa_idx) != len(eoa_idx):
            raise ValueError(f"invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}")

        range_begin = 1 if skip_first_block else 0
        sep_len = len(self.codec_tool.sep_ids)

        segments = []

        for i in range(range_begin, len(soa_idx)):
            codec_ids = ids[soa_idx[i] + 1 + sep_len: eoa_idx[i]]
            codec_ids = codec_ids[: 2 * (codec_ids.shape[0] // 2)]
            tracks = rearrange(codec_ids, "(n b) -> b n", b=2)
            segments.append(tracks.tolist())

        return segments

    def shorten_input(self, seq: torch.Tensor, max_context: int):
        # Iteratively drop the oldest segment in the context until the sequence fits in context
        pattern = torch.tensor(self.start_of_segment)
        pattern_length = pattern.numel()
        while seq.shape[-1] > max_context:
            windows = seq[0].unfold(0, pattern_length, 1)
            matches = (windows == pattern).all(dim=1)
            match_indices = torch.nonzero(matches).flatten()
            if match_indices.numel() < 3:
                # Ensure that at least one other segment remains before the current segment for continuity
                print("Unable to keep enough segments for smart context, falling back to simple truncation. " f"Now using the last {max_context} tokens.")
                return seq[:, -max_context:]
            first_segment_start = match_indices[0].item()
            second_segment_start = match_indices[1].item()
            seq = torch.cat((seq[:, :first_segment_start], seq[:, second_segment_start:]), dim=-1)
        return seq

class Stage1Pipeline_EXL2(Stage1Pipeline):

    def __init__(self, model_path: str, device: torch.device, cache_size: int, cache_mode: str, **kwargs):
        super().__init__(device, **kwargs)

        assert device != "cpu", "ExLlamaV2 does not support CPU inference."

        self.model_path = model_path

        # Define cache
        self.cache_size = cache_size
        self.cache_mode = get_cache_class(cache_mode)

    def load_model(self):
        # Load EXL2 model
        device_idx = self.device.index
        gpu_split = [0] * torch.cuda.device_count()
        gpu_split[device_idx] = 9999
        exl2_config = ExLlamaV2Config(self.model_path)
        exl2_config.no_sdpa = True  # TODO: Figure out why SDPA slows to a crawl when given custom attn mask
        self.model = ExLlamaV2(exl2_config)
        self.model.load(gpu_split)

        # Load tokenizer (only needed for vocab size in disallow_tokens)
        self.tokenizer = ExLlamaV2Tokenizer(exl2_config)

        # TODO: Output layer could be trimmed here to avoid masking out the first 32k tokens during generation

    def unload_model(self):
        self.model.unload()
        self.codec_model = None

    def generate(
        self,
        generation_token: Callable[[],bool],
        start_context: list[int],
        prompts: list[tuple[int, list[int]]],
        max_new_tokens: int,
        sample_settings: SampleSettings,
    ) -> torch.Tensor:

        if sample_settings.guidance_scale is None:
            bsz = 1
            cfg = False
            position_offsets = None
            input_mask = None
        else:
            bsz = 2
            cfg = True

        # Cache for the whole output sequence
        cache = self.cache_mode(self.model, batch_size=bsz, max_seq_len=self.cache_size)

        # Collect output here
        seq = torch.empty((bsz, 0), dtype=torch.long)

        # Sample settings
        gen_settings = ExLlamaV2Sampler.Settings(
            top_k=0, top_p=sample_settings.top_p, token_repetition_penalty=sample_settings.repetition_penalty, temperature=sample_settings.temperature
        )
        gen_settings.allow_tokens(self.tokenizer, [32002] + list(range(45334, 56722)))

        # RNG for sampling, could seed here
        rng = random.Random()

        # Prepare start context
        start_context = torch.tensor([start_context] * bsz, dtype=torch.long)
        seq = torch.cat((seq, start_context), dim=-1)

        nr_generated = 0
        progress = tqdm(total=max_new_tokens)

        # Fill cache with start context
        self.model.forward(start_context, cache=cache, preprocess_only=True)

        for iseg, segment in takewhile(lambda _: nr_generated < max_new_tokens, enumerate(prompts)):
            nr_segment_tokens, prompt_ids = segment
            prompt_ids = torch.tensor([prompt_ids] * bsz, dtype=torch.long)

            # Accept prompt tokens
            seq = torch.cat((seq, prompt_ids), dim=-1)

            # Least of remaining segment or total remaining tokens
            max_segment_tokens = min(nr_segment_tokens, max_new_tokens-nr_generated)

            # Use window slicing in case output sequence exceeds the context of model
            max_context = self.cache_size - max_segment_tokens - 1
            if seq.shape[-1] > max_context:
                print(f"Section {iseg}: output length {seq.shape[-1]} exceeding context length {max_context}, " f"dropping early segment(s) from prompt.")
                cache.current_seq_len = 0
                full_ids = self.shorten_input(seq, max_context)
                incremental_ids = full_ids
            else:
                full_ids = seq
                incremental_ids = prompt_ids

            # For the unconditional context, mask out all but the last token
            if cfg:
                mask_len = full_ids.shape[-1] - 1
                full_mask = torch.zeros((2, cache.max_seq_len), dtype=torch.half, device=self.device)
                full_mask[1, :mask_len] = -65504.0
                position_offsets = torch.tensor([[0], [-mask_len]], dtype=torch.int)
                input_mask = full_mask[:, : full_ids.shape[-1]]

            # Forward prompt
            logits = self.model.forward(incremental_ids[:, :], cache=cache, input_mask=input_mask, position_offsets=position_offsets, last_id_only=True)

            # Generate until EOS or current segment length
            for _ in range(max_segment_tokens):
                if generation_token and not generation_token():
                    break

                # Transformers-equiv. CFG
                if cfg:
                    cfg_scale = sample_settings.guidance_scale
                    logits = logits.float()
                    logits = F.log_softmax(logits, dim=-1)
                    logits = cfg_scale * logits[0] + (1 - cfg_scale) * logits[1]
                    logits = logits.unsqueeze(0)

                # Sample
                logits = logits.float().cpu()
                sample, _, _, _, _ = ExLlamaV2Sampler.sample(logits, gen_settings, full_ids[:1], rng.random(), self.tokenizer)
                if cfg:
                    sample = torch.cat((sample, sample), dim=0)

                # Accept token
                full_ids = torch.cat((full_ids, sample), dim=-1)
                seq = torch.cat((seq, sample), dim=-1)

                # Get next logits (update cache even if sample is EOA and we don't need next logits)
                if cfg:
                    input_mask = full_mask[:, : full_ids.shape[-1]]
                logits = self.model.forward(sample, cache=cache, input_mask=input_mask, position_offsets=position_offsets)

                progress.update()

                # End on EOA
                if sample[0].item() == self.mmtokenizer.eoa:
                    break

                # Don't count EOA
                nr_generated = nr_generated + 1

            # Make sure sequence ends with EOA if we reached segment_length
            else:
                sample = torch.tensor([[self.mmtokenizer.eoa]] * bsz, dtype=torch.long)
                seq = torch.cat((seq, sample), dim=-1)
                # Update cache with forced token
                self.model.forward(sample, cache=cache)

            if generation_token and not generation_token():
                break

        progress.close()

        del cache
        torch.cuda.empty_cache()
        gc.collect()

        raw_output = seq[:1, :]
        return raw_output
