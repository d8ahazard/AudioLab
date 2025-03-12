@echo off
echo Running system diagnostics for ONNX Runtime GPU compatibility...

REM Activate virtual environment
call .venv\Scripts\activate

echo =============== Python Environment ===============
python --version
pip --version

echo =============== CUDA Information ===============
where nvcc
if %ERRORLEVEL% EQU 0 (
    nvcc --version
) else (
    echo CUDA compiler not found in PATH
)

REM Check CUDA with PyTorch
echo =============== PyTorch CUDA Detection ===============
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo =============== Environment Variables ===============
echo CUDA_PATH = %CUDA_PATH%
echo PATH entries containing CUDA:
echo %PATH% | findstr /i "cuda"

echo =============== ONNX Runtime Detection ===============
python -c "import sys; print('Python executable:', sys.executable)"
python -c "import site; print('Site packages:', site.getsitepackages())"
python -c "try: import onnxruntime as ort; print(f'ONNX Runtime version: {ort.__version__}'); print(f'Available providers: {ort.get_available_providers()}'); print('GPU is available!' if 'CUDAExecutionProvider' in ort.get_available_providers() else 'GPU is NOT available') except Exception as e: print(f'Error importing ONNX Runtime: {e}')"

echo =============== DLL Debugging ===============
echo Checking for ONNX Runtime DLLs in package location...
python -c "import os, site; site_packages = site.getsitepackages()[0]; print('Looking in:', site_packages); ort_paths = [os.path.join(site_packages, 'onnxruntime'), os.path.join(site_packages, 'onnxruntime_gpu')]; [print(f'Contents of {path}:') or [print(f' - {f}') for f in os.listdir(path)] if os.path.exists(path) else print(f'{path} does not exist') for path in ort_paths]"

echo =============== Visual C++ Redistributable Check ===============
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Version || echo "Visual C++ 2015-2022 Redistributable may not be installed"

echo =============== DirectML Check ===============
python -c "try: import onnxruntime as ort; print('DirectMLExecutionProvider available!' if 'DirectMLExecutionProvider' in ort.get_available_providers() else 'DirectMLExecutionProvider NOT available') except Exception as e: print(f'Error checking DirectML: {e}')"

echo Done! Please review the above information to diagnose ONNX Runtime issues.
pause 