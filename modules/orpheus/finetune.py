"""
Orpheus TTS fine-tuning utilities for AudioLab.
"""

import os
import logging
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union

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
        self.repo_dir = os.path.join(self.model_dir, "Orpheus-TTS")
        
        # Create necessary directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Check if the repository exists
        if not os.path.exists(self.repo_dir):
            logger.error(f"Orpheus-TTS repository not found at {self.repo_dir}. Please run the setup script first.")
            raise FileNotFoundError(f"Orpheus-TTS repository not found at {self.repo_dir}. Please run the setup script first.")
        
        # Check if required packages are installed
        try:
            import orpheus_tts
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
                
        # Check other required packages
        try:
            import transformers
            import datasets
            import torch
        except ImportError as e:
            logger.error(f"Required package not installed: {e}. Please run the setup script first.")
            raise ImportError(f"Required package not installed: {e}. Please run the setup script first.")
    
    def prepare_dataset(self, audio_dir: str, speaker_name: str) -> str:
        """
        Prepare a dataset for fine-tuning from a directory of audio files.
        
        Args:
            audio_dir: Directory containing audio files (.mp3, .wav, etc.)
            speaker_name: Name to give the speaker/voice
            
        Returns:
            Path to the prepared dataset directory
        """
        import datetime
        import torchaudio
        from datasets import Dataset, Audio
        
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
    
    def prepare_training_config(self, dataset_dir: str, speaker_name: str, 
                              base_model: str = "canopylabs/orpheus-tts-0.1-pretrained",
                              training_args: Dict = None) -> str:
        """
        Prepare configuration for fine-tuning.
        
        Args:
            dataset_dir: Path to the transcribed dataset directory
            speaker_name: Name of the speaker/voice
            base_model: Base model to fine-tune
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
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        training_dir = os.path.join(self.output_dir, f"{speaker_name.lower()}_{timestamp}")
        os.makedirs(training_dir, exist_ok=True)
        
        # Create config file
        config = {
            "base_model": base_model,
            "dataset_dir": dataset_dir,
            "output_dir": training_dir,
            "speaker_name": speaker_name,
            "training_args": default_args
        }
        
        config_path = os.path.join(training_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Training configuration prepared and saved to {config_path}")
        return config_path
    
    def run_finetune(self, config_path: str, progress_callback=None) -> str:
        """
        Run fine-tuning process using the prepared configuration.
        
        Args:
            config_path: Path to the training configuration file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the fine-tuned model
        """
        # Load configuration
        with open(config_path, "r") as f:
            config = json.load(f)
        
        base_model = config["base_model"]
        dataset_dir = config["dataset_dir"]
        output_dir = config["output_dir"]
        speaker_name = config["speaker_name"]
        training_args = config["training_args"]
        
        logger.info(f"Starting fine-tuning process for {speaker_name}")
        logger.info(f"Base model: {base_model}")
        logger.info(f"Dataset: {dataset_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Set environment variables for model directories
        os.environ["HF_HOME"] = self.model_dir
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.model_dir, "transformers")
        
        # Create a training script in the output directory
        train_script = os.path.join(output_dir, "train.py")
        
        with open(train_script, "w") as f:
            f.write("""
import os
import json
import logging
import torch
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

base_model = config["base_model"]
dataset_dir = config["dataset_dir"]
output_dir = config["output_dir"]
speaker_name = config["speaker_name"]
training_args = config["training_args"]

# Load dataset
logger.info(f"Loading dataset from {dataset_dir}")
dataset = load_from_disk(dataset_dir)

# Split dataset
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Load model and tokenizer
logger.info(f"Loading base model: {base_model}")
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16 if training_args.get("fp16", False) else torch.float32,
    device_map="auto"
)

# Define training arguments
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=training_args.get("batch_size", 1),
    per_device_eval_batch_size=training_args.get("batch_size", 1),
    gradient_accumulation_steps=training_args.get("gradient_accumulation_steps", 4),
    learning_rate=training_args.get("learning_rate", 5e-5),
    num_train_epochs=training_args.get("num_train_epochs", 3),
    save_steps=training_args.get("save_steps", 500),
    save_total_limit=training_args.get("save_total_limit", 3),
    logging_steps=training_args.get("logging_steps", 100),
    evaluation_strategy=training_args.get("evaluation_strategy", "steps"),
    eval_steps=training_args.get("eval_steps", 500),
    warmup_steps=training_args.get("warmup_steps", 100),
    weight_decay=training_args.get("weight_decay", 0.01),
    fp16=training_args.get("fp16", True),
    load_best_model_at_end=True,
    report_to="wandb" if os.environ.get("WANDB_DISABLED") != "true" else "none",
    run_name=f"orpheus-finetune-{speaker_name}"
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Train model
logger.info("Starting training...")
trainer.train()

# Save final model
logger.info(f"Saving model to {output_dir}")
trainer.save_model()
tokenizer.save_pretrained(output_dir)

logger.info("Training complete!")
""")
        
        # Execute the training script
        try:
            cmd = [sys.executable, train_script]
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
                                    total_epochs = float(training_args.get("num_train_epochs", 3))
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
            return output_dir
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            raise