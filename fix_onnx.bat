@echo off
echo Fixing ONNX Runtime GPU issues...

REM Activate virtual environment
call .venv\Scripts\activate

REM Set environment variables for ONNX Runtime
SET ORT_STRATEGY=CUDA_LATEST
SET GPU_FORCE_64_BIT_PTR=1
setx ORT_STRATEGY CUDA_LATEST
setx GPU_FORCE_64_BIT_PTR 1

REM Set CUDA path and add to PATH
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
setx PATH "%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"

REM Reinstall critical dependencies
python -m pip install numpy==1.26.4 scipy==1.15.0 protobuf==3.19.6 setuptools==69.0.3 --force-reinstall

REM Remove existing ONNX Runtime installations
echo Removing existing ONNX Runtime installations...
python -m pip uninstall onnxruntime onnxruntime-gpu -y

REM Verify CUDA availability
echo Checking CUDA availability...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

REM Install ONNX Runtime GPU
echo Installing ONNX Runtime GPU v1.14.1...
python -m pip install onnxruntime-gpu==1.14.1

REM Test ONNX Runtime
echo Testing if ONNX Runtime works with GPU...
python -c "import onnxruntime as ort; print(f'ONNX Runtime version: {ort.__version__}'); print(f'Available providers: {ort.get_available_providers()}'); print('GPU is available!' if 'CUDAExecutionProvider' in ort.get_available_providers() else 'GPU is NOT available')"

echo Done! If you see "GPU is available!" above, the fix was successful.
echo Try running your application now.
pause 