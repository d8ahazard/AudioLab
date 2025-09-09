#!/usr/bin/env python
import platform
import onnx
import onnx2torch
import onnxruntime as ort
from audio_separator.separator.architectures.mdx_separator import MDXSeparator

og_load_model = None


def patched_load_model(self):
    """
    Load the ONNX model and initialize it with ONNXRuntime using optimal execution providers.
    Prioritizes TensorRT > CUDA > CPU based on availability.
    """
    try:
        self.logger.debug("Loading ONNX model for inference with TensorRT and CUDA priority...")

        if self.segment_size == self.dim_t:
            ort_session_options = ort.SessionOptions()

            # Check available ONNXRuntime providers
            available_providers = ort.get_available_providers()
            self.logger.debug(f"Available ONNX providers: {available_providers}")

            # Default to CPU in case no GPU providers are available
            providers = [("CPUExecutionProvider", {})]

            # Add CUDA if available
            if "CUDAExecutionProvider" in available_providers:
                providers.insert(0, ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}))

            # Add TensorRT if available
            if "TensorrtExecutionProvider" in available_providers:
                providers.insert(0, ("TensorrtExecutionProvider", {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": ".caches",
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": ".caches",
                    "trt_builder_optimization_level": 5,
                }))

            self.logger.debug(f"Using ONNXRuntime providers: {providers}")

            # Create ONNXRuntime inference session
            ort_inference_session = ort.InferenceSession(
                self.model_path, providers=[p[0] if isinstance(p, tuple) else p for p in providers],
                provider_options=[p[1] for p in providers if isinstance(p, tuple)],
                sess_options=ort_session_options
            )

            self.model_run = lambda spek: ort_inference_session.run(None, {"input": spek.cpu().numpy()})[0]
            self.logger.debug("Model loaded successfully using ONNXRuntime inference session.")

        else:
            if platform.system() == "Windows":
                onnx_model = onnx.load(self.model_path)
                self.model_run = onnx2torch.convert(onnx_model)
            else:
                self.model_run = onnx2torch.convert(self.model_path)

            self.model_run.to(self.torch_device).eval()
            self.logger.warning(
                "Model converted from ONNX to PyTorch due to segment size mismatch. Performance may be affected.")
    except:
        self.logger.error("Failed to load ONNX model with optimal execution providers. Falling back to CPU only.")
        self.model_run = None


# Monkey-patch MDXSeparator
def patch_separator():
    global og_load_model
    if og_load_model is not None:
        print("Already patched")
        return
    print("Patching MDXSeparator.load_model")
    og_load_model = MDXSeparator.load_model
    MDXSeparator.load_model = patched_load_model
