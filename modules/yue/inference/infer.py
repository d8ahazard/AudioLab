import copy
import os
import re
import uuid
from collections import Counter
from typing import Callable

import numpy as np
import torch
import torchaudio
from einops import rearrange
from omegaconf import OmegaConf
from torchaudio.transforms import Resample
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

from handlers.config import model_path, output_path
from modules.yue.inference.codecmanipulator import CodecManipulator
from modules.yue.inference.mmtokenizer import _MMSentencePieceTokenizer
from modules.yue.inference.xcodec_mini_infer.models.soundstream_hubert_new import SoundStream # DO NOT REMOVE THIS IMPORT
from modules.yue.inference.xcodec_mini_infer.post_process_audio import replace_low_freq_with_energy_matched
from modules.yue.inference.xcodec_mini_infer.vocoder import build_codec_model

codectool = None
codectool_stage2 = None
mmtokenizer = None
device = None


class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores


def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)
    # Convert to mono
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio


def split_lyrics(lyrics):
    pattern = r"\[(\w+)\](.*?)\n(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
    return structured_lyrics


def stage2_generate(model, prompt, batch_size=16):
    global codectool, mmtokenizer, device
    if not isinstance(codectool, CodecManipulator):
        raise ValueError("Please set the codec tool first.")
    if not isinstance(mmtokenizer, _MMSentencePieceTokenizer):
        raise ValueError("Please set the tokenizer first.")
    codec_ids = codectool.unflatten(prompt, n_quantizer=1)
    codec_ids = codectool.offset_tok_ids(
        codec_ids,
        global_offset=codectool.global_offset,
        codebook_size=codectool.codebook_size,
        num_codebooks=codectool.num_codebooks,
    ).astype(np.int32)

    # Prepare prompt_ids based on batch size or single input
    if batch_size > 1:
        codec_list = []
        for i in range(batch_size):
            idx_begin = i * 300
            idx_end = (i + 1) * 300
            codec_list.append(codec_ids[:, idx_begin:idx_end])

        codec_ids = np.concatenate(codec_list, axis=0)
        prompt_ids = np.concatenate(
            [
                np.tile([mmtokenizer.soa, mmtokenizer.stage_1], (batch_size, 1)),
                codec_ids,
                np.tile([mmtokenizer.stage_2], (batch_size, 1)),
            ],
            axis=1
        )
    else:
        prompt_ids = np.concatenate([
            np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
            codec_ids.flatten(),  # Flatten the 2D array to 1D
            np.array([mmtokenizer.stage_2])
        ]).astype(np.int32)
        prompt_ids = prompt_ids[np.newaxis, ...]

    codec_ids = torch.as_tensor(codec_ids).to(device)
    prompt_ids = torch.as_tensor(prompt_ids).to(device)
    len_prompt = prompt_ids.shape[-1]

    block_list = LogitsProcessorList(
        [BlockTokenRangeProcessor(0, 46358), BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size)])

    # Teacher forcing generate loop
    for frames_idx in range(codec_ids.shape[1]):
        cb0 = codec_ids[:, frames_idx:frames_idx + 1]
        prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
        input_ids = prompt_ids

        with torch.no_grad():
            stage2_output = model.generate(input_ids=input_ids,
                                           min_new_tokens=7,
                                           max_new_tokens=7,
                                           eos_token_id=mmtokenizer.eoa,
                                           pad_token_id=mmtokenizer.eoa,
                                           logits_processor=block_list,
                                           )

        assert stage2_output.shape[1] - prompt_ids.shape[
            1] == 7, f"output new tokens={stage2_output.shape[1] - prompt_ids.shape[1]}"
        prompt_ids = stage2_output

    # Return output based on batch size
    if batch_size > 1:
        output = prompt_ids.cpu().numpy()[:, len_prompt:]
        output_list = [output[i] for i in range(batch_size)]
        output = np.concatenate(output_list, axis=0)
    else:
        output = prompt_ids[0].cpu().numpy()[len_prompt:]

    return output


def stage2_inference(model, stage1_output_set, stage2_output_dir, batch_size=4):
    global codectool_stage2
    if not isinstance(codectool_stage2, CodecManipulator):
        raise ValueError("Please set the codec tool first.")
    stage2_result = []
    for i in tqdm(range(len(stage1_output_set))):
        output_filename = os.path.join(stage2_output_dir, os.path.basename(stage1_output_set[i]))

        if os.path.exists(output_filename):
            print(f'{output_filename} stage2 has done.')
            continue

        # Load the prompt
        prompt = np.load(stage1_output_set[i]).astype(np.int32)

        # Only accept 6s segments
        output_duration = prompt.shape[-1] // 50 // 6 * 6
        num_batch = output_duration // 6

        if num_batch <= batch_size:
            # If num_batch is less than or equal to batch_size, we can infer the entire prompt at once
            output = stage2_generate(model, prompt[:, :output_duration * 50], batch_size=num_batch)
        else:
            # If num_batch is greater than batch_size, process in chunks of batch_size
            segments = []
            num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)

            for seg in range(num_segments):
                start_idx = seg * batch_size * 300
                # Ensure the end_idx does not exceed the available length
                end_idx = min((seg + 1) * batch_size * 300, output_duration * 50)  # Adjust the last segment
                current_batch_size = batch_size if seg != num_segments - 1 or num_batch % batch_size == 0 else num_batch % batch_size
                segment = stage2_generate(
                    model,
                    prompt[:, start_idx:end_idx],
                    batch_size=current_batch_size
                )
                segments.append(segment)

            # Concatenate all the segments
            output = np.concatenate(segments, axis=0)

        # Process the ending part of the prompt
        if output_duration * 50 != prompt.shape[-1]:
            ending = stage2_generate(model, prompt[:, output_duration * 50:], batch_size=1)
            output = np.concatenate([output, ending], axis=0)
        output = codectool_stage2.ids2npy(output)

        # Fix invalid codes (a dirty solution, which may harm the quality of audio)
        # We are trying to find better one
        fixed_output = copy.deepcopy(output)
        for idx, line in enumerate(output):
            for j, element in enumerate(line):
                if element < 0 or element > 1023:
                    counter = Counter(line)
                    most_frequent = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                    fixed_output[idx, j] = most_frequent
        # save output
        np.save(output_filename, fixed_output)
        stage2_result.append(output_filename)
    return stage2_result


# convert audio tokens to audio
def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)


def generate_music(
        stage1_model: str,
        stage2_model: str,
        genre_txt: str,
        lyrics_txt: str,
        use_audio_prompt: bool = False,
        audio_prompt_path: str = "",
        prompt_start_time: float = 0.0,
        prompt_end_time: float = 30.0,
        max_new_tokens: int = 3000,
        run_n_segments: int = 2,
        stage2_batch_size: int = 4,
        keep_intermediate: bool = False,
        disable_offload_model: bool = False,
        cuda_idx: int = 0,
        rescale: bool = False,
        # You can expose these typical generation hyperparams as well:
        top_p: float = 0.93,
        temperature: float = 1.0,
        repetition_penalty: float = 1.2,
        callback: Callable = None
):
    """
    Generates multi-stage music from text prompts, optional audio prompts, and uses
    stage1 / stage2 models that reside under `os.path.join(model_path, "YuE")`.

    Args:
        stage1_model (str): Path to Stage 1 model.
        stage2_model (str): Path to Stage 2 model.
        genre_txt (str): A string containing the genre tags of the music.
        lyrics_txt (str): A string containing the lyrics of the music.
        use_audio_prompt (bool): Whether to include an audio reference as prompt.
        audio_prompt_path (str): Path to audio prompt file (used if `use_audio_prompt` is True).
        prompt_start_time (float): Start time (seconds) of the audio prompt slice.
        prompt_end_time (float): End time (seconds) of the audio prompt slice.
        max_new_tokens (int): Max tokens per generation (stage1).
        run_n_segments (int): How many lyric segments to run.
        stage2_batch_size (int): Batch size used in Stage 2.
        keep_intermediate (bool): Whether to retain intermediate artifacts (unused in sample).
        disable_offload_model (bool): If True, don't offload the Stage 1 model from GPU to CPU.
        cuda_idx (int): Which GPU to use (if any).
        rescale (bool): Rescale final audio or not (avoid clipping).
        top_p (float): Nucleus sampling hyperparameter.
        temperature (float): Sampling temperature.
        repetition_penalty (float): Penalty for repeated tokens.
        callback (Callable): Optional callback function for progress updates.

    Returns:
        A list of the paths to the output files
    """

    global codectool, codectool_stage2, mmtokenizer, device

    if use_audio_prompt and not audio_prompt_path:
        raise FileNotFoundError(
            "Please provide `audio_prompt_path` when `use_audio_prompt` is True."
        )

    # ------------------------------------------------------------------
    # 1) Model paths (assume everything is under model_path/YuE):
    # ------------------------------------------------------------------

    basic_model_config = os.path.join(model_path, "YuE", "config.yaml")
    resume_path = os.path.join(model_path, "YuE", "ckpt_00360000.pth")
    config_path = os.path.join(model_path, "YuE", "config_decoder.yaml")
    vocal_decoder_path = os.path.join(model_path, "YuE", "decoder_131000.pth")
    inst_decoder_path = os.path.join(model_path, "YuE", "decoder_151000.pth")

    # ------------------------------------------------------------------
    # 2) Output paths (always under output_path/YuE):
    # ------------------------------------------------------------------
    base_out_dir = os.path.join(output_path, "YuE")
    stage1_output_dir = os.path.join(base_out_dir, "stage1")
    stage2_output_dir = os.path.join(base_out_dir, "stage2")

    os.makedirs(stage1_output_dir, exist_ok=True)
    os.makedirs(stage2_output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3) Load device, tokenizer, Stage1 model, and xcodec model
    # ------------------------------------------------------------------
    device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
    mmtokenizer = _MMSentencePieceTokenizer(
        os.path.join(model_path, "YuE", "tokenizer.model")
    )

    # Load Stage1 model
    model = AutoModelForCausalLM.from_pretrained(
        stage1_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to(device)
    model.eval()

    # Set up CodecManipulators for stage1 and stage2
    codectool = CodecManipulator("xcodec", 0, 1)
    codectool_stage2 = CodecManipulator("xcodec", 0, 8)

    # Load the xcodec base model
    model_config = OmegaConf.load(basic_model_config)
    codec_class_name = model_config.generator.name
    codec_model = SoundStream(**model_config.generator.config).to(device)
    # Eval is bad and scares people, plus import inspection is a thing
    # codec_model = eval(codec_class_name)(**model_config.generator.config).to(device)
    parameter_dict = torch.load(resume_path, map_location="cpu")
    codec_model.load_state_dict(parameter_dict["codec_model"])
    codec_model.to(device)
    codec_model.eval()

    # ------------------------------------------------------------------
    # 4) Prepare Stage1 prompting
    # ------------------------------------------------------------------
    from modules.yue.inference.xcodec_mini_infer.vocoder import process_audio
    genres = genre_txt
    lyrics = split_lyrics(lyrics_txt)

    full_lyrics = "\n".join(lyrics)
    prompt_texts = [
        f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"
    ]
    prompt_texts += lyrics

    random_id = uuid.uuid4()
    output_seq = None

    # special tokens
    start_of_segment = mmtokenizer.tokenize("[start_of_segment]")
    end_of_segment = mmtokenizer.tokenize("[end_of_segment]")

    # Additional placeholders
    raw_output = None

    # If we're not using audio_prompt, we can unload the codec_model
    if not use_audio_prompt:
        print("Offloading codec model from GPU to CPU...")
        codec_model.to("cpu")
        codec_model.eval()
        torch.cuda.empty_cache()

    # We generate N segments in total. The first prompt is instructions, so skip it in iteration
    run_n_segments = min(run_n_segments + 1, len(lyrics))
    for i, p in enumerate(tqdm(prompt_texts[:run_n_segments])):
        section_text = p.replace("[start_of_segment]", "").replace("[end_of_segment]", "")
        guidance_scale = 1.5 if i <= 1 else 1.2
        if i == 0:
            continue

        if i == 1:
            if use_audio_prompt:
                # Load user-supplied audio prompt
                audio_prompt = load_audio_mono(audio_prompt_path)
                audio_prompt.unsqueeze_(0)
                with torch.no_grad():
                    raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=0.5)
                raw_codes = raw_codes.transpose(0, 1).cpu().numpy().astype(np.int16)
                # Format audio prompt
                code_ids = codectool.npy2ids(raw_codes[0])
                audio_prompt_codec = code_ids[int(prompt_start_time * 50): int(prompt_end_time * 50)]
                audio_prompt_codec_ids = [mmtokenizer.soa] + codectool.sep_ids + audio_prompt_codec + [
                    mmtokenizer.eoa
                ]
                sentence_ids = mmtokenizer.tokenize(
                    "[start_of_reference]") + audio_prompt_codec_ids + mmtokenizer.tokenize("[end_of_reference]")
                head_id = mmtokenizer.tokenize(prompt_texts[0]) + sentence_ids
            else:
                head_id = mmtokenizer.tokenize(prompt_texts[0])

            prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [
                mmtokenizer.soa] + codectool.sep_ids
        else:
            prompt_ids = (
                    end_of_segment
                    + start_of_segment
                    + mmtokenizer.tokenize(section_text)
                    + [mmtokenizer.soa]
                    + codectool.sep_ids
            )

        prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device)

        if i > 1:
            input_ids = torch.cat([raw_output, prompt_ids], dim=1)
        else:
            input_ids = prompt_ids

        # Check context limit
        max_context = 16384 - max_new_tokens - 1
        if input_ids.shape[-1] > max_context:
            print(
                f"Section {i}: input length {input_ids.shape[-1]} > context length {max_context},"
                " truncating to last {max_context} tokens."
            )
            input_ids = input_ids[:, -max_context:]

        # Create block processor for restricting token ranges

        with torch.no_grad():
            gen_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=100,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=LogitsProcessorList(
                    [
                        BlockTokenRangeProcessor(0, 32002),
                        BlockTokenRangeProcessor(32016, 32016),
                    ]
                ),
                guidance_scale=guidance_scale,
            )
            # Ensure it ends with [eoa]
            if gen_output[0][-1].item() != mmtokenizer.eoa:
                tensor_eoa = torch.as_tensor([[mmtokenizer.eoa]], device=device)
                gen_output = torch.cat((gen_output, tensor_eoa), dim=1)

        if i > 1:
            # append new result
            new_part = gen_output[:, input_ids.shape[-1]:]
            raw_output = torch.cat([raw_output, prompt_ids, new_part], dim=1)
        else:
            raw_output = gen_output

    if use_audio_prompt:
        print("Offloading codec model from GPU to CPU...")
        codec_model.to("cpu")
        codec_model.eval()
        torch.cuda.empty_cache()
    # ------------------------------------------------------------------
    # 5) Parse out vocal vs. instrumental from the final Stage1 output
    # ------------------------------------------------------------------
    ids = raw_output[0].cpu().numpy()
    soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
    eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
    if len(soa_idx) != len(eoa_idx):
        raise ValueError(
            f"Invalid pairs of soa/eoa. #soa: {len(soa_idx)}, #eoa: {len(eoa_idx)}"
        )

    vocals = []
    instrumentals = []
    range_begin = 1 if use_audio_prompt else 0
    for i in range(range_begin, len(soa_idx)):
        codec_ids = ids[soa_idx[i] + 1: eoa_idx[i]]
        if codec_ids[0] == 32016:
            codec_ids = codec_ids[1:]
        # ensure even length
        codec_ids = codec_ids[: 2 * (codec_ids.shape[0] // 2)]
        vocals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
        instrumentals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])

        vocals.append(vocals_ids)
        instrumentals.append(instrumentals_ids)

    vocals = np.concatenate(vocals, axis=1)
    instrumentals = np.concatenate(instrumentals, axis=1)

    # Save stage1 outputs
    vocal_save_path = os.path.join(
        stage1_output_dir,
        f"cot_{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_vocal_{random_id}".replace(
            ".", "@"
        )
        + ".npy",
    )
    inst_save_path = os.path.join(
        stage1_output_dir,
        f"cot_{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_instrumental_{random_id}".replace(
            ".", "@"
        )
        + ".npy",
    )

    np.save(vocal_save_path, vocals)
    np.save(inst_save_path, instrumentals)

    stage1_output_set = [vocal_save_path, inst_save_path]

    # Offload Stage1 model if desired
    if not disable_offload_model:
        print("Offloading model from GPU to CPU...")
        model.cpu()
        del model
        model = None
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 6) Stage2 Inference
    # ------------------------------------------------------------------
    print("Stage 2 inference...")

    model_stage2 = AutoModelForCausalLM.from_pretrained(
        stage2_model,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    model_stage2.to(device)
    model_stage2.eval()

    stage2_result = stage2_inference(
        model_stage2, stage1_output_set, stage2_output_dir, batch_size=stage2_batch_size
    )
    print(stage2_result)
    print("Stage 2 DONE.\n")

    # Unload stage2 model
    print("Offloading Stage2 model from GPU to CPU...")
    model_stage2.to("cpu")
    model_stage2.eval()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 7) Reconstruct .wav from codes
    # ------------------------------------------------------------------
    recons_output_dir = os.path.join(base_out_dir, "recons")
    recons_mix_dir = os.path.join(recons_output_dir, "mix")
    os.makedirs(recons_mix_dir, exist_ok=True)

    def save_audio(wav: torch.Tensor, path, sample_rate: int, do_rescale: bool = False):
        folder_path = os.path.dirname(path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        limit = 0.99
        max_val = wav.abs().max()
        wav = wav * min(limit / max_val, 1) if do_rescale else wav.clamp(-limit, limit)

        # Ensure the file is saved as .wav
        wav_path = os.path.splitext(path)[0] + ".wav"
        print(f"Saving audio to {wav_path}")
        # Save as WAV instead of MP3
        torchaudio.save(wav_path, wav, sample_rate=sample_rate, encoding="PCM_S", bits_per_sample=16)

        return wav_path  # Return the correct WAV path

    tracks = []
    # Reload codec model here
    if not use_audio_prompt:
        codec_model.to(device)
        codec_model.eval()

    for npy in stage2_result:
        codec_result = np.load(npy)
        with torch.no_grad():
            decoded_waveform = codec_model.decode(
                torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long)
                .unsqueeze(0)
                .permute(1, 0, 2)
                .to(device)
            )
        decoded_waveform = decoded_waveform.cpu().squeeze(0)
        save_path = os.path.join(
            recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".wav"
        )
        tracks.append(save_path)
        save_audio(decoded_waveform, save_path, 16000, do_rescale=rescale)

    # Mix vocals & instrumentals
    import soundfile as sf
    for inst_path in tracks:
        try:
            if inst_path.endswith(".wav") and "instrumental" in inst_path:
                # Attempt to find matching vocal track
                vocal_path = inst_path.replace("instrumental", "vocal")
                if not os.path.exists(vocal_path):
                    continue
                recons_mix = os.path.join(
                    recons_mix_dir, os.path.basename(inst_path).replace("instrumental", "mixed")
                )
                vocal_stem, sr = sf.read(inst_path)
                instrumental_stem, _ = sf.read(vocal_path)
                mix_stem = (vocal_stem + instrumental_stem) / 1
                sf.write(recons_mix, mix_stem, sr)
        except Exception as e:
            print(e)


    # ------------------------------------------------------------------
    # 8) Final upsampling (Vocoder)
    # ------------------------------------------------------------------
    # Set this here so it's always got something...
    vocoder_output_dir = os.path.join(base_out_dir, "vocoder")
    vocoder_stems_dir = os.path.join(vocoder_output_dir, "stems")
    vocoder_mix_dir = os.path.join(vocoder_output_dir, "mix")
    final_vocal_path = os.path.join(vocoder_stems_dir, "vocal.wav"),
    final_inst_path = os.path.join(vocoder_stems_dir, "instrumental.wav")
    recons_mix = os.path.join(recons_mix_dir, "mixed_upsampled.wav")
    final_path = os.path.join(base_out_dir, os.path.basename(recons_mix))

    os.makedirs(vocoder_mix_dir, exist_ok=True)
    os.makedirs(vocoder_stems_dir, exist_ok=True)

    vocal_output = None
    instrumental_output = None

    # Pre-loaded from above: vocal_decoder, inst_decoder
    vocal_decoder, inst_decoder = build_codec_model(config_path, vocal_decoder_path, inst_decoder_path)

    for npy in stage2_result:
        if "instrumental" in npy:
            # Process instrumental
            instrumental_output = process_audio(
                npy,
                final_inst_path,
                rescale,
                None,  # We no longer pass 'args' but pass needed arguments directly
                inst_decoder,
                codec_model,
            )
        else:
            # Process vocal
            vocal_output = process_audio(
                npy,
                final_vocal_path,
                rescale,
                None,
                vocal_decoder,
                codec_model,
            )

        try:
            mix_output = instrumental_output + vocal_output

            save_audio(mix_output, recons_mix, 44100, rescale)
            print(f"Created upsampled mix: {recons_mix}")

            # Post-process
            replace_low_freq_with_energy_matched(
                a_file=recons_mix,  # 16kHz in original code, but we used 44.1k just now
                b_file=recons_mix,  # If you have a separate 48k or 44.1k mix, you might adjust
                c_file=final_path,
                cutoff_freq=5500.0,
            )
            return [final_path, final_vocal_path, final_inst_path]
        except RuntimeError as e:
            print(e)
            print(f"Mixing or post-process failed for {recons_mix}!")

    # Delete all models after moving to CPU
    for cleanup in [codec_model, vocal_decoder, inst_decoder, model, model_stage2]:
        try:
            cleanup.to("cpu")
            del cleanup
        except Exception as e:
            print(f"Error unloading model: {e}")
    torch.cuda.empty_cache()


#
# def main():
#     global codectool, codectool_stage2, mmtokenizer, device
#     parser = argparse.ArgumentParser()
#     # Model Configuration:
#     parser.add_argument("--stage1_model", type=str, default="m-a-p/YuE-s1-7B-anneal-en-cot",
#                         help="The model checkpoint path or identifier for the Stage 1 model.")
#     parser.add_argument("--stage2_model", type=str, default="m-a-p/YuE-s2-1B-general",
#                         help="The model checkpoint path or identifier for the Stage 2 model.")
#     parser.add_argument("--max_new_tokens", type=int, default=3000,
#                         help="The maximum number of new tokens to generate in one pass during text generation.")
#     parser.add_argument("--run_n_segments", type=int, default=2,
#                         help="The number of segments to process during the generation.")
#     parser.add_argument("--stage2_batch_size", type=int, default=4, help="The batch size used in Stage 2 inference.")
#     # Prompt
#     parser.add_argument("--genre_txt", type=str, required=True,
#                         help="The file path to a text file containing genre tags that describe the musical style or characteristics (e.g., instrumental, genre, mood, vocal timbre, vocal gender). This is used as part of the generation prompt.")
#     parser.add_argument("--lyrics_txt", type=str, required=True,
#                         help="The file path to a text file containing the lyrics for the music generation. These lyrics will be processed and split into structured segments to guide the generation process.")
#     parser.add_argument("--use_audio_prompt", action="store_true",
#                         help="If set, the model will use an audio file as a prompt during generation. The audio file should be specified using --audio_prompt_path.")
#     parser.add_argument("--audio_prompt_path", type=str, default="",
#                         help="The file path to an audio file to use as a reference prompt when --use_audio_prompt is enabled.")
#     parser.add_argument("--prompt_start_time", type=float, default=0.0,
#                         help="The start time in seconds to extract the audio prompt from the given audio file.")
#     parser.add_argument("--prompt_end_time", type=float, default=30.0,
#                         help="The end time in seconds to extract the audio prompt from the given audio file.")
#     # Output
#     parser.add_argument("--output_dir", type=str, default="./output",
#                         help="The directory where generated outputs will be saved.")
#     parser.add_argument("--keep_intermediate", action="store_true",
#                         help="If set, intermediate outputs will be saved during processing.")
#     parser.add_argument("--disable_offload_model", action="store_true",
#                         help="If set, the model will not be offloaded from the GPU to CPU after Stage 1 inference.")
#     parser.add_argument("--cuda_idx", type=int, default=0)
#     # Config for xcodec and upsampler
#     parser.add_argument('--basic_model_config', default='./xcodec_mini_infer/final_ckpt/config.yaml',
#                         help='YAML files for xcodec configurations.')
#     parser.add_argument('--resume_path', default='./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth',
#                         help='Path to the xcodec checkpoint.')
#     parser.add_argument('--config_path', type=str, default='./xcodec_mini_infer/decoders/config.yaml',
#                         help='Path to Vocos config file.')
#     parser.add_argument('--vocal_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_131000.pth',
#                         help='Path to Vocos decoder weights.')
#     parser.add_argument('--inst_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_151000.pth',
#                         help='Path to Vocos decoder weights.')
#     parser.add_argument('-r', '--rescale', action='store_true', help='Rescale output to avoid clipping.')
#
#     args = parser.parse_args()
#     if args.use_audio_prompt and not args.audio_prompt_path:
#         raise FileNotFoundError(
#             "Please offer audio prompt filepath using '--audio_prompt_path', when you enable 'use_audio_prompt'!")
#     stage1_model = args.stage1_model
#     stage2_model = args.stage2_model
#     cuda_idx = args.cuda_idx
#     max_new_tokens = args.max_new_tokens
#     stage1_output_dir = os.path.join(args.output_dir, f"stage1")
#     stage2_output_dir = stage1_output_dir.replace('stage1', 'stage2')
#     os.makedirs(stage1_output_dir, exist_ok=True)
#     os.makedirs(stage2_output_dir, exist_ok=True)
#
#     # load tokenizer and model
#     device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
#     mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")
#     model = AutoModelForCausalLM.from_pretrained(
#         stage1_model,
#         torch_dtype=torch.bfloat16,
#         attn_implementation="flash_attention_2",  # To enable flashattn, you have to install flash-attn
#     )
#     # to device, if gpu is available
#     model.to(device)
#     model.eval()
#
#     codectool = CodecManipulator("xcodec", 0, 1)
#     codectool_stage2 = CodecManipulator("xcodec", 0, 8)
#     model_config = OmegaConf.load(args.basic_model_config)
#     codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
#     parameter_dict = torch.load(args.resume_path, map_location='cpu')
#     codec_model.load_state_dict(parameter_dict['codec_model'])
#     codec_model.to(device)
#     codec_model.eval()
#
#     # Call the function and print the result
#     stage1_output_set = []
#     # Tips:
#     # genre tags support instrumental，genre，mood，vocal timbr and vocal gender
#     # all kinds of tags are needed
#     with open(args.genre_txt) as f:
#         genres = f.read().strip()
#     with open(args.lyrics_txt) as f:
#         lyrics = split_lyrics(f.read())
#     # intruction
#     full_lyrics = "\n".join(lyrics)
#     prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
#     prompt_texts += lyrics
#
#     random_id = uuid.uuid4()
#     output_seq = None
#     # Here is suggested decoding config
#     top_p = 0.93
#     temperature = 1.0
#     repetition_penalty = 1.2
#     # special tokens
#     start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
#     end_of_segment = mmtokenizer.tokenize('[end_of_segment]')
#     # Format text prompt
#     run_n_segments = min(args.run_n_segments + 1, len(lyrics))
#     for i, p in enumerate(tqdm(prompt_texts[:run_n_segments])):
#         section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
#         guidance_scale = 1.5 if i <= 1 else 1.2
#         if i == 0:
#             continue
#         if i == 1:
#             if args.use_audio_prompt:
#                 audio_prompt = load_audio_mono(args.audio_prompt_path)
#                 audio_prompt.unsqueeze_(0)
#                 with torch.no_grad():
#                     raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=0.5)
#                 raw_codes = raw_codes.transpose(0, 1)
#                 raw_codes = raw_codes.cpu().numpy().astype(np.int16)
#                 # Format audio prompt
#                 code_ids = codectool.npy2ids(raw_codes[0])
#                 audio_prompt_codec = code_ids[int(args.prompt_start_time * 50): int(
#                     args.prompt_end_time * 50)]  # 50 is tps of xcodec
#                 audio_prompt_codec_ids = [mmtokenizer.soa] + codectool.sep_ids + audio_prompt_codec + [mmtokenizer.eoa]
#                 sentence_ids = mmtokenizer.tokenize(
#                     "[start_of_reference]") + audio_prompt_codec_ids + mmtokenizer.tokenize(
#                     "[end_of_reference]")
#                 head_id = mmtokenizer.tokenize(prompt_texts[0]) + sentence_ids
#             else:
#                 head_id = mmtokenizer.tokenize(prompt_texts[0])
#             prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [
#                 mmtokenizer.soa] + codectool.sep_ids
#         else:
#             prompt_ids = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [
#                 mmtokenizer.soa] + codectool.sep_ids
#
#         prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device)
#         input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids
#         # Use window slicing in case output sequence exceeds the context of model
#         max_context = 16384 - max_new_tokens - 1
#         if input_ids.shape[-1] > max_context:
#             print(
#                 f'Section {i}: output length {input_ids.shape[-1]} exceeding context length {max_context}, now using the last {max_context} tokens.')
#             input_ids = input_ids[:, -max_context:]
#         with torch.no_grad():
#             output_seq = model.generate(
#                 input_ids=input_ids,
#                 max_new_tokens=max_new_tokens,
#                 min_new_tokens=100,
#                 do_sample=True,
#                 top_p=top_p,
#                 temperature=temperature,
#                 repetition_penalty=repetition_penalty,
#                 eos_token_id=mmtokenizer.eoa,
#                 pad_token_id=mmtokenizer.eoa,
#                 logits_processor=LogitsProcessorList(
#                     [BlockTokenRangeProcessor(0, 32002), BlockTokenRangeProcessor(32016, 32016)]),
#                 guidance_scale=guidance_scale,
#             )
#             if output_seq[0][-1].item() != mmtokenizer.eoa:
#                 tensor_eoa = torch.as_tensor([[mmtokenizer.eoa]]).to(model.device)
#                 output_seq = torch.cat((output_seq, tensor_eoa), dim=1)
#         if i > 1:
#             raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1]:]], dim=1)
#         else:
#             raw_output = output_seq
#
#     # save raw output and check sanity
#     ids = raw_output[0].cpu().numpy()
#     soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
#     eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
#     if len(soa_idx) != len(eoa_idx):
#         raise ValueError(f'invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}')
#
#     vocals = []
#     instrumentals = []
#     range_begin = 1 if args.use_audio_prompt else 0
#     for i in range(range_begin, len(soa_idx)):
#         codec_ids = ids[soa_idx[i] + 1:eoa_idx[i]]
#         if codec_ids[0] == 32016:
#             codec_ids = codec_ids[1:]
#         codec_ids = codec_ids[:2 * (codec_ids.shape[0] // 2)]
#         vocals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
#         vocals.append(vocals_ids)
#         instrumentals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
#         instrumentals.append(instrumentals_ids)
#     vocals = np.concatenate(vocals, axis=1)
#     instrumentals = np.concatenate(instrumentals, axis=1)
#     vocal_save_path = os.path.join(stage1_output_dir,
#                                    f"cot_{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_vocal_{random_id}".replace(
#                                        '.', '@') + '.npy')
#     inst_save_path = os.path.join(stage1_output_dir,
#                                   f"cot_{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_instrumental_{random_id}".replace(
#                                       '.', '@') + '.npy')
#     np.save(vocal_save_path, vocals)
#     np.save(inst_save_path, instrumentals)
#     stage1_output_set.append(vocal_save_path)
#     stage1_output_set.append(inst_save_path)
#
#     # offload model
#     if not args.disable_offload_model:
#         model.cpu()
#         del model
#         torch.cuda.empty_cache()
#
#     print("Stage 2 inference...")
#     model_stage2 = AutoModelForCausalLM.from_pretrained(
#         stage2_model,
#         torch_dtype=torch.float16,
#         attn_implementation="flash_attention_2"
#     )
#     model_stage2.to(device)
#     model_stage2.eval()
#
#     stage2_result = stage2_inference(model_stage2, stage1_output_set, stage2_output_dir,
#                                      batch_size=args.stage2_batch_size)
#     print(stage2_result)
#     print('Stage 2 DONE.\n')
#
#     # reconstruct tracks
#     recons_output_dir = os.path.join(args.output_dir, "recons")
#     recons_mix_dir = os.path.join(recons_output_dir, 'mix')
#     os.makedirs(recons_mix_dir, exist_ok=True)
#     tracks = []
#     for npy in stage2_result:
#         codec_result = np.load(npy)
#         decodec_rlt = []
#         with torch.no_grad():
#             decoded_waveform = codec_model.decode(
#                 torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(
#                     device))
#         decoded_waveform = decoded_waveform.cpu().squeeze(0)
#         decodec_rlt.append(torch.as_tensor(decoded_waveform))
#         decodec_rlt = torch.cat(decodec_rlt, dim=-1)
#         save_path = os.path.join(recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".mp3")
#         tracks.append(save_path)
#         save_audio(decodec_rlt, save_path, 16000)
#     # mix tracks
#     for inst_path in tracks:
#         try:
#             if (inst_path.endswith('.wav') or inst_path.endswith('.mp3')) \
#                     and 'instrumental' in inst_path:
#                 # find pair
#                 vocal_path = inst_path.replace('instrumental', 'vocal')
#                 if not os.path.exists(vocal_path):
#                     continue
#                 # mix
#                 recons_mix = os.path.join(recons_mix_dir, os.path.basename(inst_path).replace('instrumental', 'mixed'))
#                 vocal_stem, sr = sf.read(inst_path)
#                 instrumental_stem, _ = sf.read(vocal_path)
#                 mix_stem = (vocal_stem + instrumental_stem) / 1
#                 sf.write(recons_mix, mix_stem, sr)
#         except Exception as e:
#             print(e)
#
#     # vocoder to upsample audios
#     vocal_decoder, inst_decoder = build_codec_model(args.config_path, args.vocal_decoder_path, args.inst_decoder_path)
#     vocoder_output_dir = os.path.join(args.output_dir, 'vocoder')
#     vocoder_stems_dir = os.path.join(vocoder_output_dir, 'stems')
#     vocoder_mix_dir = os.path.join(vocoder_output_dir, 'mix')
#     os.makedirs(vocoder_mix_dir, exist_ok=True)
#     os.makedirs(vocoder_stems_dir, exist_ok=True)
#     for npy in stage2_result:
#         if 'instrumental' in npy:
#             # Process instrumental
#             instrumental_output = process_audio(
#                 npy,
#                 os.path.join(vocoder_stems_dir, 'instrumental.mp3'),
#                 args.rescale,
#                 args,
#                 inst_decoder,
#                 codec_model
#             )
#         else:
#             # Process vocal
#             vocal_output = process_audio(
#                 npy,
#                 os.path.join(vocoder_stems_dir, 'vocal.mp3'),
#                 args.rescale,
#                 args,
#                 vocal_decoder,
#                 codec_model
#             )
#     # mix tracks
#     try:
#         mix_output = instrumental_output + vocal_output
#         vocoder_mix = os.path.join(vocoder_mix_dir, os.path.basename(recons_mix))
#         save_audio(mix_output, vocoder_mix, 44100, args.rescale)
#         print(f"Created mix: {vocoder_mix}")
#     except RuntimeError as e:
#         print(e)
#         print(f"mix {vocoder_mix} failed! inst: {instrumental_output.shape}, vocal: {vocal_output.shape}")
#
#     # Post process
#     replace_low_freq_with_energy_matched(
#         a_file=recons_mix,  # 16kHz
#         b_file=vocoder_mix,  # 48kHz
#         c_file=os.path.join(args.output_dir, os.path.basename(recons_mix)),
#         cutoff_freq=5500.0
#     )
