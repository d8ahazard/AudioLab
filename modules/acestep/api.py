"""
ACE-Step API endpoints for AudioLab.
"""

import os
import logging
from typing import List, Optional, Any
from fastapi import Body, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from modules.acestep.process import process, process_lora, DEFAULT_MODEL
from handlers.config import output_path

# Configure logging
logger = logging.getLogger("ADLB.ACEStep.API")

# API Models
class GenerateRequest(BaseModel):
    """Request model for text-to-music generation"""
    prompt: str = Field(..., description="Text prompt describing the music to generate")
    lyrics: str = Field("", description="Lyrics for the vocals (optional)")
    audio_duration: float = Field(60.0, description="Duration of generated audio in seconds")
    model_name: str = Field(DEFAULT_MODEL, description="Name of the ACE-Step model to use")
    bf16: bool = Field(True, description="Whether to use bfloat16 precision")
    torch_compile: bool = Field(False, description="Whether to use torch compile for faster inference")
    cpu_offload: bool = Field(False, description="Whether to offload models to CPU to save VRAM")
    overlapped_decode: bool = Field(False, description="Whether to use overlapped decoding")
    device_id: int = Field(0, description="GPU device ID to use")
    infer_step: int = Field(27, description="Number of inference steps")
    guidance_scale: float = Field(7.5, description="Guidance scale for generation")
    scheduler_type: str = Field("flow_match_euler", description="Scheduler type (flow_match_euler, euler, dpm)")
    cfg_type: str = Field("apg", description="CFG type (apg, cfg)")
    omega_scale: float = Field(10.0, description="Omega scale parameter")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

class GenerateResponse(BaseModel):
    """Response model for text-to-music generation"""
    status: str = Field(..., description="Status of the request (success or error)")
    audio_path: Optional[str] = Field(None, description="Path to the generated audio file")
    message: str = Field(..., description="Status message")

class LoRAGenerateRequest(BaseModel):
    """Request model for LoRA-based music generation"""
    prompt: str = Field(..., description="Text prompt describing the music to generate")
    lyrics: str = Field(..., description="Lyrics for the vocals (required for rap generation)")
    audio_duration: float = Field(60.0, description="Duration of generated audio in seconds")
    base_model_name: str = Field(DEFAULT_MODEL, description="Name of the base ACE-Step model")
    lora_model_path: str = Field(..., description="Path to the LoRA model weights")
    bf16: bool = Field(True, description="Whether to use bfloat16 precision")
    torch_compile: bool = Field(False, description="Whether to use torch compile for faster inference")
    cpu_offload: bool = Field(False, description="Whether to offload models to CPU to save VRAM")
    device_id: int = Field(0, description="GPU device ID to use")
    infer_step: int = Field(27, description="Number of inference steps")
    guidance_scale: float = Field(7.5, description="Guidance scale for generation")
    scheduler_type: str = Field("flow_match_euler", description="Scheduler type")
    cfg_type: str = Field("apg", description="CFG type")
    omega_scale: float = Field(10.0, description="Omega scale parameter")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

class LoRAGenerateResponse(BaseModel):
    """Response model for LoRA-based music generation"""
    status: str = Field(..., description="Status of the request (success or error)")
    audio_path: Optional[str] = Field(None, description="Path to the generated audio file")
    message: str = Field(..., description="Status message")

def register_api_endpoints(api):
    """
    Register ACE-Step API endpoints.
    
    Args:
        api: FastAPI application instance
    """
    @api.post("/api/v1/acestep/generate", tags=["Audio Generation"], response_model=GenerateResponse)
    async def generate_audio(request: GenerateRequest = Body(...)):
        """
        Generate music using ACE-Step.
        
        This endpoint uses the ACE-Step model to generate music based on provided parameters.
        
        Returns:
            JSON response with generated audio path and status information
        """
        try:
            # Validate prompt
            if not request.prompt or request.prompt.strip() == "":
                raise HTTPException(status_code=400, detail="Prompt is required")
            
            # Run generation process
            output_path, message = process(
                prompt=request.prompt,
                lyrics=request.lyrics,
                audio_duration=request.audio_duration,
                model_name=request.model_name,
                bf16=request.bf16,
                torch_compile=request.torch_compile,
                cpu_offload=request.cpu_offload,
                overlapped_decode=request.overlapped_decode,
                device_id=request.device_id,
                infer_step=request.infer_step,
                guidance_scale=request.guidance_scale,
                scheduler_type=request.scheduler_type,
                cfg_type=request.cfg_type,
                omega_scale=request.omega_scale,
                seed=request.seed
            )
            
            # Check if generation was successful
            if output_path and os.path.exists(output_path):
                return GenerateResponse(
                    status="success",
                    audio_path=output_path,
                    message=message
                )
            else:
                return GenerateResponse(
                    status="error",
                    audio_path=None,
                    message=message or "Failed to generate audio"
                )
                
        except Exception as e:
            logger.error(f"Error in ACE-Step generate endpoint: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @api.post("/api/v1/acestep/lora/generate", tags=["Audio Generation"], response_model=LoRAGenerateResponse)
    async def generate_lora_audio(request: LoRAGenerateRequest = Body(...)):
        """
        Generate music using ACE-Step with a LoRA model.
        
        This endpoint uses ACE-Step with a specialized LoRA model like RapMachine.
        
        Returns:
            JSON response with generated audio path and status information
        """
        try:
            # Validate prompt and lyrics
            if not request.prompt or request.prompt.strip() == "":
                raise HTTPException(status_code=400, detail="Prompt is required")
                
            if not request.lyrics or request.lyrics.strip() == "":
                raise HTTPException(status_code=400, detail="Lyrics are required for LoRA models")
            
            # Run LoRA generation process
            output_path, message = process_lora(
                prompt=request.prompt,
                lyrics=request.lyrics,
                audio_duration=request.audio_duration,
                base_model_name=request.base_model_name,
                lora_model_path=request.lora_model_path,
                bf16=request.bf16,
                torch_compile=request.torch_compile,
                cpu_offload=request.cpu_offload,
                device_id=request.device_id,
                infer_step=request.infer_step,
                guidance_scale=request.guidance_scale,
                scheduler_type=request.scheduler_type,
                cfg_type=request.cfg_type,
                omega_scale=request.omega_scale,
                seed=request.seed
            )
            
            # Check if generation was successful
            if output_path and os.path.exists(output_path):
                return LoRAGenerateResponse(
                    status="success",
                    audio_path=output_path,
                    message=message
                )
            else:
                return LoRAGenerateResponse(
                    status="error",
                    audio_path=None,
                    message=message or "Failed to generate audio with LoRA model"
                )
                
        except Exception as e:
            logger.error(f"Error in ACE-Step LoRA generate endpoint: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e)) 