import copy
from collections import Counter
from collections.abc import Callable
import gc
import math
import os

import numpy as np
import torch
from modules.yue.source.yue.common import get_cache_class
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
from modules.yue.source.yue.mmtokenizer import _MMSentencePieceTokenizer
from tqdm import tqdm

def align(n, m):
    return ((n + m - 1) // m) * m

def split_bsz(bsz, maxbsz):
    n_sub_batches = math.ceil(bsz / maxbsz)
    base_size = bsz // n_sub_batches
    remainder = bsz % n_sub_batches
    sub_batch_sizes = [base_size + 1] * remainder + [base_size] * (n_sub_batches - remainder)
    indices = []
    start = 0
    for size in sub_batch_sizes:
        end = start + size
        indices.append((start, end))
        start = end
    return indices

class Stage2Pipeline:

    def __init__(self, device: torch.device):
        self.device = device

        # Load tokenizer
        self.mmtokenizer = _MMSentencePieceTokenizer(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mm_tokenizer_v0.2_hf", "tokenizer.model"))

    def fix_output(self, output):
        # Fix invalid codes (a dirty solution, which may harm the quality of audio)
        # We are trying to find better one
        fixed_output = copy.deepcopy(output)
        for i, line in enumerate(output):
            for j, element in enumerate(line):
                if element < 0 or element > 1023:
                    counter = Counter(line)
                    most_frequant = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                    fixed_output[i, j] = most_frequant
        return fixed_output

class Stage2Pipeline_EXL2(Stage2Pipeline):

    def __init__(self, model_path: str, device: torch.device, cache_size: int, cache_mode: str):
        super().__init__(device)

        self.cache_size = cache_size

        assert device != "cpu", "ExLlamaV2 does not support CPU inference."

        # Load EXL2 model
        device_idx = self.device.index
        gpu_split = [0] * torch.cuda.device_count()
        gpu_split[device_idx] = 9999
        exl2_config = ExLlamaV2Config(model_path)
        self.model = ExLlamaV2(exl2_config)
        self.model.load(gpu_split)

        # Move embedding layer to GPU to avoid CPU sync during argmax gen loop
        self.model.modules[0].device_idx = self.model.modules[1].device_idx
        self.model.modules[0].reload()

        # Load tokenizer (only needed for vocab size in disallow_tokens)
        self.tokenizer = ExLlamaV2Tokenizer(exl2_config)

        # Define cache
        self.cache_mode = get_cache_class(cache_mode)

    def unload_model(self):
        self.model.unload()

    def generate(self,
                 generation_token: Callable[[],bool],
                 stage1_tracks: list[list[int], list[int]],
                 stage2_tracks: list[list[int], list[int]],
                 ) -> list[np.array]:

        full_batch = []

        for itrack, track in tqdm(enumerate(stage1_tracks)):
            prompt = torch.as_tensor([track], dtype=torch.long)
            segs = torch.split(prompt, 300, dim=-1)

            for seg_idx, seg in enumerate(segs):
                seg_len = seg.shape[-1]
                full_batch.append((seg_len, seg_idx, itrack, seg))

        # Prepare segments
        prefix = torch.tensor([[self.mmtokenizer.soa, self.mmtokenizer.stage_1]], dtype=torch.long)
        suffix = torch.tensor([[self.mmtokenizer.stage_2]], dtype=torch.long)
        for i in range(len(full_batch)):
            seg_len, seg_idx, output_idx, codec_ids = full_batch[i]
            prompt_ids = torch.cat((prefix, codec_ids, suffix), dim=-1)
            full_batch[i] = (seg_len, seg_idx, output_idx, codec_ids, prompt_ids)

        # Group prompts by length
        batches = {}
        for seq in full_batch:
            if not seq[0] in batches:
                batches[seq[0]] = []
            batches[seq[0]].append(seq)

        # Split into on minibatches
        split_batch = []
        for idx, (seg_len, batch) in enumerate(batches.items()):
            b_seg_order = [b[1] for b in batch]
            b_part_order = [b[2] for b in batch]
            b_codec_ids = torch.cat([b[3] for b in batch], dim=0)
            b_prompt_ids = torch.cat([b[4] for b in batch], dim=0)

            max_bsz = self.cache_size // align(b_prompt_ids.shape[1] + b_codec_ids.shape[1] * 8, 32)
            assert max_bsz > 0
            for a, b in split_bsz(b_prompt_ids.shape[0], max_bsz):
                split_batch.append((b_seg_order[a:b], b_part_order[a:b], b_codec_ids[a:b], b_prompt_ids[a:b]))

        # Inference
        output_parts = []
        for _ in stage2_tracks:
            output_parts.append([])

        for seg_order, part_order, codec_ids, prompt_ids in tqdm(split_batch):
            codec_ids = codec_ids.to(self.device)
            prompt_ids = prompt_ids.to(self.device)
            batch_size, len_prompt = prompt_ids.shape

            cache = self.cache_mode(self.model, batch_size=batch_size, max_seq_len=align(prompt_ids.shape[1] + codec_ids.shape[1] * 8, 32))
            output_ids = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)

            for frames_idx in tqdm(range(codec_ids.shape[1])):
                if generation_token and not generation_token():
                    break
                cb0 = codec_ids[:, frames_idx : frames_idx + 1]

                # Append the initial prompt to the first codec frame
                if frames_idx == 0:
                    cb0 = torch.cat([prompt_ids, cb0], dim=-1)

                # Forward prompt
                output_ids = torch.cat((output_ids, cb0), dim=-1)
                logits = self.model.forward(cb0, cache=cache, last_id_only=True)

                for i in range(7):

                    # Slice logits instead of biasing start and end of distribution
                    first_logit = 46358
                    last_logit = 53526
                    logits = logits[:, :, first_logit:last_logit]

                    # Greedy sampling
                    sample = logits.argmax(dim=-1) + first_logit
                    output_ids = torch.cat((output_ids, sample), dim=-1)

                    # TODO: Here, original asserts that we didn't sample mmtokenizer.eoa (can we just mask it out?)

                    # Forward sample
                    logits = self.model.forward(sample, cache=cache)

            # Release cache tensors
            del cache
            torch.cuda.empty_cache()
            gc.collect()

            if generation_token and not generation_token():
                break

            # Trim prompt
            output_ids = output_ids[:, len_prompt:]

            # Split outputs
            for i in range(batch_size):
                output_parts[part_order[i]].append((seg_order[i], output_ids[i : i + 1, :]))


        if generation_token and not generation_token():
            return None

        # Unshuffle and recombine output parts
        output_tracks = []
        for i, p in enumerate(output_parts):
            p = sorted(p, key=lambda x: x[0])
            part_o = torch.cat([pp[1] for pp in p], dim=-1).flatten().cpu().numpy()
            # Add cached tokens
            output = stage2_tracks[i]
            if part_o.shape[0] > 0:
                output = np.concatenate((output, part_o.tolist()))
            output_tracks.append(output)

        return output_tracks
