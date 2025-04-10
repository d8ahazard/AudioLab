# Copyright (c) 2025 ASLP-LAB
#               2025 Ziqian Ning   (ningziqian@mail.nwpu.edu.cn)
#               2025 Huakang Chen  (huakang@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib.resources import files
from handlers.config import app_path, model_path, output_path

from modules.diffrythm.model.cfm import CFM
from modules.diffrythm.model.dit import DiT
from modules.diffrythm.model.trainer import Trainer

import json
import os
import torch
from dataclasses import dataclass

os.environ['OMP_NUM_THREADS']="1"
os.environ['MKL_NUM_THREADS']="1"

@dataclass
class TrainingArgs:
    """Arguments for DiffRhythm training"""
    # Required parameters
    project_dir: str  # Path to preprocessed data directory
    base_model: str   # Base model to start from
    
    # Training parameters with defaults matching UI
    batch_size: int = 8
    epochs: int = 110
    learning_rate: float = 7.5e-5
    num_workers: int = 4
    save_steps: int = 5000
    warmup_steps: int = 20
    max_grad_norm: float = 1.0
    grad_accumulation: int = 1
    
    # Dropout probabilities
    audio_drop_prob: float = 0.3
    style_drop_prob: float = 0.1
    lrc_drop_prob: float = 0.1
    
    # Model parameters
    max_frames: int = 2048
    grad_ckpt: bool = False
    reset_lr: bool = False
    
    # Other parameters
    resumable_with_seed: int = 42

def train(args: TrainingArgs, progress=None):
    """Train a DiffRhythm model with the given arguments
    
    Args:
        args: Training arguments
        progress: Optional gradio progress callback
    """
    
    if progress:
        progress(0.1, "Loading base model configuration...")
        
    # Load base model config
    model_config_path = os.path.join(model_path, args.base_model, "config.json")
    with open(model_config_path) as f:
        model_config = json.load(f)

    if progress:
        progress(0.2, "Initializing model...")
        
    # Initialize model
    model_cls = DiT
    model = CFM(
        transformer=model_cls(**model_config["model"], max_frames=args.max_frames),
        num_channels=model_config["model"]['mel_dim'],
        audio_drop_prob=args.audio_drop_prob,
        cond_drop_prob=0.1,  # Fixed value from original implementation
        style_drop_prob=args.style_drop_prob,
        lrc_drop_prob=args.lrc_drop_prob,
        max_frames=args.max_frames
    )

    if progress:
        progress(0.3, "Loading base model weights...")
        
    # Load base model weights if they exist
    base_model_path = os.path.join(model_path, args.base_model, "model.pth")
    if os.path.exists(base_model_path):
        state_dict = torch.load(base_model_path, map_location="cpu")
        model.load_state_dict(state_dict["model_state_dict"], strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    if progress:
        progress(0.4, "Initializing trainer...")
        
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=args,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_warmup_updates=args.warmup_steps,
        save_per_updates=args.save_steps,
        checkpoint_path=os.path.join(args.project_dir, "checkpoints"),
        grad_accumulation_steps=args.grad_accumulation,
        max_grad_norm=args.max_grad_norm,
        wandb_project=f"diffrythm-{os.path.basename(args.project_dir)}",
        wandb_run_name=os.path.basename(args.project_dir),
        wandb_resume_id=None,
        last_per_steps=args.save_steps * args.grad_accumulation,
        bnb_optimizer=False,
        reset_lr=args.reset_lr,
        batch_size=args.batch_size,
        grad_ckpt=args.grad_ckpt
    )

    if progress:
        progress(0.5, "Starting training...")
        
    # Start training
    trainer.train(resumable_with_seed=args.resumable_with_seed)
    
    if progress:
        progress(0.9, "Saving final model...")
    
    # Save final model
    final_model_path = os.path.join(model_path, "diffrythm_custom", f"{os.path.basename(args.project_dir)}.pth")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    
    # Save model state
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_config
    }, final_model_path)
    
    if progress:
        progress(1.0, "Training complete!")
    
    return final_model_path

def main():
    """
    Main entry point - this should not be called directly.
    Use the train() function instead which takes TrainingArgs.
    """
    raise NotImplementedError(
        "Do not call main() directly. Use train() with TrainingArgs instead."
    )
