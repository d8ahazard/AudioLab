import logging
import os
import re
import uuid
import copy
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
from modules.yue.inference.xcodec_mini_infer.models.soundstream_hubert_new import SoundStream
from modules.yue.inference.xcodec_mini_infer.post_process_audio import replace_low_freq_with_energy_matched
from modules.yue.inference.xcodec_mini_infer.vocoder import build_codec_model
from modules.yue.inference.xcodec_mini_infer.vocoder import process_audio

logger = logging.getLogger("ADLB")


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
    """Split lyrics into sections based on [section] markers.
    
    Args:
        lyrics (str): Input lyrics with sections marked like [verse], [chorus], etc.
        
    Returns:
        list: List of formatted section strings
    """
    # Add a newline at the end if not present to help with regex
    if not lyrics.endswith('\n'):
        lyrics += '\n'
        
    # Match sections including their content up to the next section or end
    pattern = r'\[(\w+)\]([\s\S]*?)(?=\[|$)'
    segments = re.findall(pattern, lyrics)
    
    # Format each section with proper spacing
    structured_lyrics = []
    for section, content in segments:
        # Clean up the content: remove extra whitespace but preserve line breaks
        cleaned_content = '\n'.join(line.strip() for line in content.strip().split('\n'))
        formatted_section = f'[{section}]\n{cleaned_content}\n\n'
        structured_lyrics.append(formatted_section)
        
    print(f"Structured lyrics: {structured_lyrics}")
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


def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)


def stage2_inference(model, stage1_output_set, stage2_output_dir, batch_size=4, callback=None, step=0, total=1):
    global codectool_stage2
    if not isinstance(codectool_stage2, CodecManipulator):
        raise ValueError("Please set the codec tool first.")

    stage2_result = []
    for i in tqdm(range(len(stage1_output_set))):
        output_filename = os.path.join(stage2_output_dir, os.path.basename(stage1_output_set[i]))

        if os.path.exists(output_filename):
            print(f'{output_filename} stage2 has done.')
            # Even if it's skipped, we might still want to mark this step done.
            stage2_result.append(output_filename)
            if callback is not None:
                step += 1
                callback(step / total, f"Stage2 inference skipped for item {i + 1}/{len(stage1_output_set)}", total)
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
        fixed_output = copy.deepcopy(output)
        for idx, line in enumerate(output):
            for j, element in enumerate(line):
                if element < 0 or element > 1023:
                    counter = Counter(line)
                    most_frequent = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                    fixed_output[idx, j] = most_frequent

        np.save(output_filename, fixed_output)
        stage2_result.append(output_filename)

        # Update progress for this item
        if callback is not None:
            step += 1
            callback(step / total, f"Stage2 inference complete for item {i + 1}/{len(stage1_output_set)}", total)

    return stage2_result, step


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
        top_p: float = 0.93,
        temperature: float = 1.0,
        repetition_penalty: float = 1.2,
        callback: Callable = None
) -> dict[str, str]:
    """
    Generates multi-stage music from text prompts and optional audio prompts.

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
        callback (Callable): Optional callback function for progress updates. Uses the same inputs as a gr.Progress(),
        i.e. progress(step, desc, total)
        
    Returns:
        dict[str, str]: Dictionary with keys "final", "vocal", and "instrumental" 
                      containing the paths to the respective output files
    """

    global codectool, codectool_stage2, mmtokenizer, device

    if use_audio_prompt and not audio_prompt_path:
        raise FileNotFoundError(
            "Please provide `audio_prompt_path` when `use_audio_prompt` is True."
        )

    # ---------------------------------------------------
    # Setup steps for the progress bar
    # We'll assume 2 stage1 outputs (vocal + instrumental).
    # total_steps = run_n_segments (Stage1 segments)
    #              + 2            (Stage2 items)
    #              + 7            (major pipeline steps)
    # ---------------------------------------------------
    total_steps = run_n_segments + 2 + 7
    step = 0

    # Dictionary to track our outputs at each stage
    outputs = {
        "vocal": None,
        "instrumental": None,
        "final": None
    }

    # ------------------------------------------------------------------
    # 1) Model paths (assume everything is under model_path/YuE):
    # ------------------------------------------------------------------
    basic_model_config = os.path.join(model_path, "YuE", "config.yaml")
    resume_path = os.path.join(model_path, "YuE", "ckpt_00360000.pth")
    config_path = os.path.join(model_path, "YuE", "config_decoder.yaml")
    vocal_decoder_path = os.path.join(model_path, "YuE", "decoder_131000.pth")
    inst_decoder_path = os.path.join(model_path, "YuE", "decoder_151000.pth")

    # ------------------------------------------------------------------
    # 2) Define all output paths upfront
    # ------------------------------------------------------------------
    # Base directories
    base_out_dir = os.path.join(output_path, "YuE")
    
    # Stage 1 directories and files
    stage1_output_dir = os.path.join(base_out_dir, "stage1")
    os.makedirs(stage1_output_dir, exist_ok=True)
    
    # Stage 2 directories and files  
    stage2_output_dir = os.path.join(base_out_dir, "stage2")
    os.makedirs(stage2_output_dir, exist_ok=True)
    
    # Reconstruction directories and files
    recons_output_dir = os.path.join(base_out_dir, "recons")
    recons_stems_dir = os.path.join(recons_output_dir, "stems")
    recons_mix_dir = os.path.join(recons_output_dir, "mix")
    os.makedirs(recons_stems_dir, exist_ok=True)
    os.makedirs(recons_mix_dir, exist_ok=True)
    
    # Final vocoder directories and files
    vocoder_output_dir = os.path.join(base_out_dir, "vocoder")
    vocoder_stems_dir = os.path.join(vocoder_output_dir, "stems")
    vocoder_mix_dir = os.path.join(vocoder_output_dir, "mix")
    os.makedirs(vocoder_stems_dir, exist_ok=True)
    os.makedirs(vocoder_mix_dir, exist_ok=True)
    
    # Generate a unique ID for this generation
    random_id = uuid.uuid4()

    # ------------------------------------------------------------------
    # 3) Load device, tokenizer, Stage1 model, and xcodec model
    # ------------------------------------------------------------------
    device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
    mmtokenizer = _MMSentencePieceTokenizer(
        os.path.join(model_path, "YuE", "tokenizer.model")
    )

    model = AutoModelForCausalLM.from_pretrained(
        stage1_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to(device)
    model.eval()

    # Update progress: loaded Stage1
    step += 1
    if callback is not None:
        callback(step / total_steps, "Loaded Stage1 model", total_steps)

    codectool = CodecManipulator("xcodec", 0, 1)
    codectool_stage2 = CodecManipulator("xcodec", 0, 8)

    model_config = OmegaConf.load(basic_model_config)
    codec_model = SoundStream(**model_config.generator.config).to(device)
    parameter_dict = torch.load(resume_path, map_location="cpu", weights_only=False)
    codec_model.load_state_dict(parameter_dict["codec_model"])
    codec_model.to(device)
    codec_model.eval()

    # ------------------------------------------------------------------
    # 4) Prepare Stage1 prompting
    # ------------------------------------------------------------------
    genres = genre_txt
    lyrics = split_lyrics(lyrics_txt)
    full_lyrics = "\n".join(lyrics)
    prompt_texts = [
        f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"
    ]
    prompt_texts += lyrics

    # Create a simple base name for the generated files
    clean_genre = genres.strip().replace(' ', '_')[:20]  # Limit genre length
    clean_genre = clean_genre.replace(",", "")
    base_filename = f"yue_{clean_genre}_{random_id}"
    
    # Define actual file paths for all stages based on the base_filename
    # Stage 1 NPY files
    vocal_stage1_path = os.path.join(stage1_output_dir, f"{base_filename}(vocal).npy")
    inst_stage1_path = os.path.join(stage1_output_dir, f"{base_filename}(instrumental).npy")
    
    # Stage 2 NPY files
    vocal_stage2_path = os.path.join(stage2_output_dir, f"{base_filename}(vocal).npy")
    inst_stage2_path = os.path.join(stage2_output_dir, f"{base_filename}(instrumental).npy")
    
    # Reconstructed audio WAV files
    vocal_recons_path = os.path.join(recons_stems_dir, f"{base_filename}(vocal).wav")
    inst_recons_path = os.path.join(recons_stems_dir, f"{base_filename}(instrumental).wav")
    mix_recons_path = os.path.join(recons_mix_dir, f"{base_filename}(final).wav")
    
    # Final vocoder WAV files
    vocal_final_path = os.path.join(vocoder_stems_dir, f"{base_filename}(vocal).wav")
    inst_final_path = os.path.join(vocoder_stems_dir, f"{base_filename}(instrumental).wav")
    final_mix_path = os.path.join(vocoder_mix_dir, f"{base_filename}(final).wav")

    output_seq = None
    start_of_segment = mmtokenizer.tokenize("[start_of_segment]")
    end_of_segment = mmtokenizer.tokenize("[end_of_segment]")
    raw_output = None

    # If not using an audio prompt, offload the codec_model early
    if not use_audio_prompt:
        print("Offloading codec model from GPU to CPU...")
        codec_model.to("cpu")
        codec_model.eval()
        torch.cuda.empty_cache()

    run_n_segments = min(run_n_segments + 1, len(lyrics))
    for i, p in enumerate(tqdm(prompt_texts[:run_n_segments])):
        if i == 0:
            continue

        section_text = p.replace("[start_of_segment]", "").replace("[end_of_segment]", "")
        print(f"Section text: {section_text}")
        guidance_scale = 1.5 if i <= 1 else 1.2

        if i == 1:
            if use_audio_prompt:
                audio_prompt = load_audio_mono(audio_prompt_path)
                audio_prompt.unsqueeze_(0)
                with torch.no_grad():
                    raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=0.5)
                raw_codes = raw_codes.transpose(0, 1).cpu().numpy().astype(np.int16)
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

        max_context = 16384 - max_new_tokens - 1
        if input_ids.shape[-1] > max_context:
            print(
                f"Section {i}: input length {input_ids.shape[-1]} > context length {max_context}, "
                "truncating..."
            )
            input_ids = input_ids[:, -max_context:]

        from transformers import LogitsProcessorList
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
            if gen_output[0][-1].item() != mmtokenizer.eoa:
                tensor_eoa = torch.as_tensor([[mmtokenizer.eoa]], device=device)
                gen_output = torch.cat((gen_output, tensor_eoa), dim=1)

        if i > 1:
            new_part = gen_output[:, input_ids.shape[-1]:]
            raw_output = torch.cat([raw_output, prompt_ids, new_part], dim=1)
        else:
            raw_output = gen_output

        # Update progress for this Stage1 segment
        step += 1
        if callback is not None:
            callback(step / total_steps, f"Generated Stage1 segment {i}/{run_n_segments - 1}", total_steps)

    if use_audio_prompt:
        print("Offloading codec model from GPU to CPU...")
        codec_model.eval()
        torch.cuda.empty_cache()

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
        codec_ids = codec_ids[: 2 * (codec_ids.shape[0] // 2)]
        vocals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
        instrumentals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])

        vocals.append(vocals_ids)
        instrumentals.append(instrumentals_ids)

    vocals = np.concatenate(vocals, axis=1)
    instrumentals = np.concatenate(instrumentals, axis=1)

    # Save Stage 1 output files
    np.save(vocal_stage1_path, vocals)
    np.save(inst_stage1_path, instrumentals)

    stage1_output_set = [vocal_stage1_path, inst_stage1_path]

    # Offload Stage1 model
    step += 1
    if callback is not None:
        callback(step / total_steps, "Offloading Stage1 model", total_steps)

    if not disable_offload_model:
        print("Offloading model from GPU to CPU...")
        model.cpu()
        del model
        model = None
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 6) Stage2 Inference
    # ------------------------------------------------------------------
    step += 1
    if callback is not None:
        callback(step / total_steps, "Loading Stage2 model", total_steps)

    model_stage2 = AutoModelForCausalLM.from_pretrained(
        stage2_model,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    model_stage2.to(device)
    model_stage2.eval()
    
    print("Stage 2 inference...")
    stage2_paths, step = stage2_inference(
        model_stage2,
        stage1_output_set,
        stage2_output_dir,
        batch_size=stage2_batch_size,
        callback=callback,
        step=step,
        total=total_steps
    )
    print("Stage 2 DONE.\n")

    step += 1
    if callback is not None:
        callback(step / total_steps, "Offloading Stage2 model", total_steps)

    print("Offloading Stage2 model from GPU to CPU...")
    model_stage2.to("cpu")
    model_stage2.eval()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 7) Reconstruct .wav from codes
    # ------------------------------------------------------------------
    step += 1
    if callback is not None:
        callback(step / total_steps, "Reconstructing raw .wav from codes", total_steps)

    def save_audio_file(wav: torch.Tensor, path, sample_rate: int, do_rescale: bool = False):
        folder_path = os.path.dirname(path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        limit = 0.99
        max_val = wav.abs().max()
        wav = wav * min(limit / max_val, 1) if do_rescale else wav.clamp(-limit, limit)

        print(f"Saving audio to {path}")
        torchaudio.save(path, wav, sample_rate=sample_rate, encoding="PCM_S", bits_per_sample=16)
        return path

    if not use_audio_prompt:
        codec_model.to(device)
        codec_model.eval()

    # Process and save the reconstructed audio files
    for i, npy_path in enumerate(stage2_paths):
        codec_result = np.load(npy_path)
        with torch.no_grad():
            decoded_waveform = codec_model.decode(
                torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long)
                .unsqueeze(0)
                .permute(1, 0, 2)
                .to(device)
            )
        decoded_waveform = decoded_waveform.cpu().squeeze(0)
        
        # Save to appropriate path
        if i == 0:  # Vocal
            save_audio_file(decoded_waveform, vocal_recons_path, 16000, do_rescale=rescale)
            outputs["vocal"] = vocal_recons_path
        else:  # Instrumental
            save_audio_file(decoded_waveform, inst_recons_path, 16000, do_rescale=rescale)
            outputs["instrumental"] = inst_recons_path

    # Create and save the reconstruction mix file
    if "vocal" in outputs and "instrumental" in outputs:
        try:
            import soundfile as sf
            print(f"Creating mix from {outputs['vocal']} and {outputs['instrumental']}")
            
            vocal_stem, sr = sf.read(outputs['vocal'])
            instrumental_stem, _ = sf.read(outputs['instrumental'])
            mix_stem = (vocal_stem + instrumental_stem) / 1
            
            sf.write(mix_recons_path, mix_stem, sr)
            outputs["final"] = mix_recons_path
            print(f"Created reconstruction mix: {mix_recons_path}")
        except Exception as e:
            print(f"Error creating reconstruction mix: {e}")

    # ------------------------------------------------------------------
    # 8) Final upsampling (Vocoder)
    # ------------------------------------------------------------------
    step += 1
    if callback is not None:
        callback(step / total_steps, "Upsampling final audio", total_steps)
    
    # Build the codec models for upsampling
    vocal_decoder, inst_decoder = build_codec_model(config_path, vocal_decoder_path, inst_decoder_path)

    # Process and save the final upsampled audio
    final_stem_outputs = []
    
    try:
        # Process vocal stem
        vocal_tensor = process_audio(
            stage2_paths[0],
            vocal_final_path,
            rescale,
            vocal_decoder,
            codec_model,
        )
        final_stem_outputs.append(vocal_tensor)
        outputs["vocal"] = vocal_final_path
        print(f"Created final vocal output: {vocal_final_path}")
        
        # Process instrumental stem
        instrumental_tensor = process_audio(
            stage2_paths[1],
            inst_final_path,
            rescale,
            inst_decoder,
            codec_model,
        )
        final_stem_outputs.append(instrumental_tensor)
        outputs["instrumental"] = inst_final_path
        print(f"Created final instrumental output: {inst_final_path}")
    
        # Create and save the final mix
        if len(final_stem_outputs) == 2:
            mix_output = final_stem_outputs[0] + final_stem_outputs[1]
            save_audio_file(mix_output, final_mix_path, 44100, rescale)
            outputs["final"] = final_mix_path
            print(f"Created final mix: {final_mix_path}")
    except Exception as e:
        print(f"Error in vocoder processing: {e}")
        # We already have fallbacks in the outputs dictionary from the reconstruction step

    # Cleanup models
    for cleanup in [codec_model, vocal_decoder, inst_decoder, model_stage2]:
        try:
            cleanup.to("cpu")
            del cleanup
        except Exception as e:
            print(f"Error unloading model: {e}")
    torch.cuda.empty_cache()
    
    # Return the outputs dictionary
    return outputs
