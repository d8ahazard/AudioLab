"""
Orpheus TTS fine-tuning utilities for AudioLab.
"""

import os
import logging
import json
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, Optional, List
import datetime
import tempfile
import torchaudio
from datasets import Dataset, Audio, load_dataset
import transformers
import datasets
import torch
import shutil
import orpheus_tts  # Import the official package
            
from handlers.config import model_path, output_path

logger = logging.getLogger("ADLB.Orpheus.Finetune")

class OrpheusFinetune:
    """
    Handler for fine-tuning Orpheus TTS models.
    """
    
    def __init__(self):
        """Initialize the OrpheusFinetune handler."""
        self.model_dir = os.path.join(model_path, "orpheus")
        self.output_dir = os.path.join(output_path, "orpheus_finetune")
        self.data_dir = os.path.join(output_path, "orpheus_data")
        
        # Create necessary directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Check if required packages are installed
        try:
            logger.debug("Successfully imported orpheus_tts")
        except ImportError as e:
            if "vllm._C" in str(e):
                logger.error("The vllm package is not properly installed. This is a dependency of orpheus_tts.")
                logger.error("Please make sure you have properly installed vllm==0.7.3 with the correct CUDA version.")
                logger.error("Try running the setup script again, or install manually: pip install vllm==0.7.3")
                raise ImportError("The vllm package is not properly installed. Please run the setup script again.") from e
            else:
                logger.error(f"Required package not installed: orpheus_tts. Please run the setup script first.")
                logger.error(f"Import error details: {e}")
                raise ImportError(f"Required package not installed: orpheus_tts. Please run the setup script first.") from e
                
    def prepare_dataset(self, audio_dir: str, speaker_name: str) -> str:
        """
        Prepare a dataset for fine-tuning from a directory of audio files.
        
        Args:
            audio_dir: Directory containing audio files (.mp3, .wav, etc.)
            speaker_name: Name to give the speaker/voice
            
        Returns:
            Path to the prepared dataset directory
        """
        
        logger.info(f"Preparing dataset from {audio_dir}")
        
        # Create dataset directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"{speaker_name.lower()}_{timestamp}"
        dataset_dir = os.path.join(self.data_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Scan for audio files
        audio_files = []
        for ext in [".wav", ".mp3", ".flac", ".m4a", ".ogg"]:
            audio_files.extend(list(Path(audio_dir).glob(f"*{ext}")))
        
        if not audio_files:
            raise ValueError(f"No audio files found in {audio_dir}")
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Create dataset entries
        dataset_entries = []
        
        for audio_file in audio_files:
            try:
                # We'll use an empty string as placeholder for text
                # This would ideally be replaced with actual transcriptions if available
                entry = {
                    "audio": str(audio_file),
                    "text": "",
                    "speaker": speaker_name
                }
                dataset_entries.append(entry)
            except Exception as e:
                logger.warning(f"Error processing {audio_file}: {e}")
        
        # Create Hugging Face dataset
        dataset = Dataset.from_dict({
            "audio": [entry["audio"] for entry in dataset_entries],
            "text": [entry["text"] for entry in dataset_entries],
            "speaker": [entry["speaker"] for entry in dataset_entries]
        })
        
        # Add audio feature
        dataset = dataset.cast_column("audio", Audio())
        
        # Save dataset info
        dataset_info = {
            "name": dataset_name,
            "speaker": speaker_name,
            "num_samples": len(dataset_entries),
            "created_at": timestamp,
            "audio_dir": audio_dir,
        }
        
        with open(os.path.join(dataset_dir, "info.json"), "w") as f:
            json.dump(dataset_info, f, indent=2)
        
        # Save dataset
        dataset.save_to_disk(os.path.join(dataset_dir, "dataset"))
        
        logger.info(f"Dataset prepared and saved to {dataset_dir}")
        return dataset_dir
    
    def transcribe_dataset(self, dataset_dir: str, progress_callback=None) -> str:
        """
        Transcribe audio files in the dataset using Whisper or other ASR model.
        
        Args:
            dataset_dir: Path to the prepared dataset directory
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the transcribed dataset
        """
        from datasets import load_from_disk, Audio
        import torch
        
        # Load dataset
        dataset_path = os.path.join(dataset_dir, "dataset")
        dataset = load_from_disk(dataset_path)
        
        # Load Whisper model
        try:
            from transformers import pipeline
            if torch.cuda.is_available():
                device = 0
            else:
                device = -1
                
            logger.info("Loading Whisper model for transcription...")
            transcriber = pipeline("automatic-speech-recognition", 
                                   model="openai/whisper-medium", 
                                   device=device)
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise
        
        # Function to transcribe audio
        def transcribe(example):
            audio = example["audio"]
            try:
                result = transcriber(audio["array"], sampling_rate=audio["sampling_rate"])
                return {"text": result["text"]}
            except Exception as e:
                logger.warning(f"Transcription error: {e}")
                return {"text": ""}
        
        # Apply transcription
        logger.info("Transcribing audio files...")
        total_samples = len(dataset)
        
        for i, example in enumerate(dataset):
            if progress_callback:
                progress_callback(i / total_samples)
                
            result = transcribe(example)
            dataset = dataset.add_item({
                "audio": example["audio"]["path"],
                "text": result["text"],
                "speaker": example["speaker"]
            })
            
            if i % 10 == 0:
                logger.info(f"Transcribed {i}/{total_samples} samples")
        
        # Save updated dataset
        transcribed_dataset_dir = os.path.join(dataset_dir, "transcribed")
        os.makedirs(transcribed_dataset_dir, exist_ok=True)
        dataset.save_to_disk(transcribed_dataset_dir)
        
        logger.info(f"Transcription completed and saved to {transcribed_dataset_dir}")
        return transcribed_dataset_dir
    
    def format_dataset_for_orpheus(self, dataset_dir: str, output_dir: Optional[str] = None) -> str:
        """
        Format the dataset according to the Orpheus TTS requirements.
        
        Args:
            dataset_dir: Path to the transcribed dataset directory
            output_dir: Optional output directory
            
        Returns:
            Path to the formatted dataset
        """
        from datasets import load_from_disk
        
        # Load dataset
        dataset = load_from_disk(dataset_dir)
        
        # Create output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(dataset_dir), "formatted")
            os.makedirs(output_dir, exist_ok=True)
        
        # Extract speaker name from dataset (assume all entries have the same speaker)
        speaker_name = dataset[0]["speaker"] if len(dataset) > 0 else "unknown"
        
        # Format the dataset as required by Orpheus
        formatted_data = []
        for item in dataset:
            if not item["text"]:
                continue
                
            # Format as required: {speaker_name}: {text}
            formatted_item = {
                "text": f"{speaker_name}: {item['text']}",
                "audio_path": item["audio"]["path"] if isinstance(item["audio"], dict) else item["audio"]
            }
            formatted_data.append(formatted_item)
        
        # Save as JSON (for debugging and inspection)
        with open(os.path.join(output_dir, "dataset.json"), "w") as f:
            json.dump(formatted_data, f, indent=2)
        
        # Create a new HF dataset
        formatted_dataset = Dataset.from_dict({
            "text": [item["text"] for item in formatted_data],
            "audio": [item["audio_path"] for item in formatted_data]
        })
        
        # Save the formatted dataset
        formatted_dataset.save_to_disk(os.path.join(output_dir, "dataset"))
        
        logger.info(f"Dataset formatted for Orpheus and saved to {output_dir}")
        return os.path.join(output_dir, "dataset")
    
    def prepare_training_config(self, dataset_dir: str, speaker_name: str, 
                              base_model: str = "canopylabs/orpheus-tts-0.1-pretrained",
                              use_lora: bool = False,
                              training_args: Dict = None) -> str:
        """
        Prepare configuration for fine-tuning.
        
        Args:
            dataset_dir: Path to the formatted dataset directory
            speaker_name: Name of the speaker/voice
            base_model: Base model to fine-tune
            use_lora: Whether to use LoRA for fine-tuning
            training_args: Additional training arguments
            
        Returns:
            Path to the training configuration file
        """
        # Default training arguments
        default_args = {
            "learning_rate": 5e-5,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 3,
            "save_steps": 500,
            "save_total_limit": 3,
            "logging_steps": 100,
            "evaluation_strategy": "steps",
            "eval_steps": 500,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "fp16": True
        }
        
        # Update with user-provided arguments
        if training_args:
            default_args.update(training_args)
        
        # Create training directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        training_dir = os.path.join(self.output_dir, f"{speaker_name.lower()}_{timestamp}")
        os.makedirs(training_dir, exist_ok=True)
        
        # Create config YAML file for the official Orpheus training script
        config = {
            "TTS_dataset": dataset_dir,
            "model_name": base_model,
            "epochs": default_args["num_train_epochs"],
            "batch_size": default_args["batch_size"],
            "number_processes": default_args.get("gradient_accumulation_steps", 4),
            "pad_token": 128263,
            "save_steps": default_args["save_steps"],
            "learning_rate": default_args["learning_rate"],
            "save_folder": os.path.join(training_dir, "checkpoints"),
            "project_name": f"orpheus-{speaker_name.lower()}",
            "run_name": f"{speaker_name.lower()}-{timestamp}"
        }
        
        # Create config.yaml file
        config_path = os.path.join(training_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # If using LoRA, copy the LoRA script to the training directory
        if use_lora:
            script_path = os.path.join(training_dir, "lora.py")
            og_files_dir = os.path.join(self.model_dir, "og_files")
            
            # Check if original files directory exists
            if not os.path.exists(og_files_dir):
                os.makedirs(og_files_dir, exist_ok=True)
                
            # Create lora.py if it doesn't exist (copy from modules/orpheus/og_files/lora.py)
            og_lora_path = os.path.join(os.path.dirname(__file__), "og_files", "lora.py")
            if os.path.exists(og_lora_path):
                shutil.copy(og_lora_path, script_path)
            else:
                # Recreate the lora.py file
                self._create_lora_script(script_path)
        else:
            # Copy the regular training script
            script_path = os.path.join(training_dir, "train.py")
            og_files_dir = os.path.join(self.model_dir, "og_files")
            
            # Check if original files directory exists
            if not os.path.exists(og_files_dir):
                os.makedirs(og_files_dir, exist_ok=True)
                
            # Create train.py if it doesn't exist (copy from modules/orpheus/og_files/train.py)
            og_train_path = os.path.join(os.path.dirname(__file__), "og_files", "train.py")
            if os.path.exists(og_train_path):
                shutil.copy(og_train_path, script_path)
            else:
                # Recreate the train.py file
                self._create_train_script(script_path)
        
        logger.info(f"Training configuration prepared and saved to {config_path}")
        return config_path
    
    def _create_train_script(self, script_path: str):
        """Create the training script file."""
        with open(script_path, "w") as f:
            f.write("""from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import yaml
import wandb

config_file = "config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]

model_name = config["model_name"]
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")


ds = load_dataset(dsn, split="train") 

wandb.init(project=project_name, name = run_name)

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size, 
    logging_steps=1,
    bf16=True,
    output_dir=f"./{base_repo_id}",
    report_to="wandb", 
    save_steps=save_steps,
    remove_unused_columns=True, 
    learning_rate=learning_rate,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
)

trainer.train()
""")
        
    def _create_lora_script(self, script_path: str):
        """Create the LoRA training script file."""
        with open(script_path, "w") as f:
            f.write("""from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import yaml
import wandb

config_file = "config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]

model_name = config["model_name"]
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]

lora_rank = 32
lora_alpha = 64
lora_dropout = 0.0

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")

lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj",  "o_proj", "gate_proj", "down_proj", "up_proj"],
    bias="none",
    modules_to_save=["lm_head", "embed_tokens"], # Optional to train the embeddings and lm head
    task_type="CAUSAL_LM",
    use_rslora=True,
)

model = get_peft_model(model, lora_config)

ds = load_dataset(dsn, split="train") 

wandb.init(project=project_name, name = run_name)

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size, 
    logging_steps=1,
    bf16=True,
    output_dir=f"./{base_repo_id}",
    report_to="wandb", 
    save_steps=save_steps,
    remove_unused_columns=True, 
    learning_rate=learning_rate,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
)

trainer.train()

merged_model = model.merge_and_unload()

merged_model.save_pretrained(f"./{base_repo_id}/merged")
tokenizer.save_pretrained(f"./{base_repo_id}/merged")
""")
    
    def run_finetune(self, config_path: str, use_lora: bool = False, progress_callback=None) -> str:
        """
        Run fine-tuning process using the prepared configuration.
        
        Args:
            config_path: Path to the training configuration file
            use_lora: Whether to use LoRA for fine-tuning
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the fine-tuned model
        """
        # Load configuration
        with open(config_path, "r") as f:
            if config_path.endswith(".json"):
                config = json.load(f)
            else:
                config = yaml.safe_load(f)
        
        output_dir = os.path.dirname(config_path)
        
        logger.info(f"Starting fine-tuning process")
        logger.info(f"Output directory: {output_dir}")
        
        # Set environment variables for model directories
        os.environ["HF_HOME"] = self.model_dir
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.model_dir, "transformers")
        
        # Determine which script to use
        if use_lora:
            script_file = os.path.join(output_dir, "lora.py")
        else:
            script_file = os.path.join(output_dir, "train.py")
            
        if not os.path.exists(script_file):
            raise FileNotFoundError(f"Training script not found at {script_file}")
        
        # Execute the training script
        try:
            cmd = [sys.executable, script_file]
            env = os.environ.copy()
            
            # Disable wandb if not needed
            env["WANDB_DISABLED"] = "true"
            
            logger.info(f"Executing training script: {' '.join(cmd)}")
            
            # Use subprocess to run the training script
            process = subprocess.Popen(
                cmd, 
                cwd=output_dir, 
                env=env, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                universal_newlines=True
            )
            
            # Monitor output
            for line in process.stdout:
                logger.info(line.strip())
                
                # Check for progress information and update callback
                if progress_callback and "train_loss" in line:
                    try:
                        # Extract training progress (roughly)
                        if "epoch" in line:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == "epoch":
                                    epoch = float(parts[i+1].strip(","))
                                    total_epochs = float(config.get("epochs", 3))
                                    progress = epoch / total_epochs
                                    progress_callback(progress)
                                    break
                    except Exception as e:
                        logger.warning(f"Error parsing progress: {e}")
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code != 0:
                logger.error(f"Training failed with exit code {return_code}")
                raise RuntimeError(f"Training process failed with exit code {return_code}")
            
            logger.info(f"Fine-tuning completed successfully. Model saved to {output_dir}")
            
            # Return the path to the fine-tuned model
            if use_lora:
                model_dir = os.path.join(output_dir, "checkpoints", "merged")
            else:
                model_dir = os.path.join(output_dir, "checkpoints")
                
            return model_dir
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            raise