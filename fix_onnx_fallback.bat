@echo off
echo Trying alternative versions of ONNX Runtime GPU...

REM Activate virtual environment
call .venv\Scripts\activate

REM Uninstall any existing ONNX Runtime
python -m pip uninstall onnxruntime onnxruntime-gpu -y

REM Try various combinations of ONNX Runtime GPU versions
echo ------------- Trying v1.13.1 -------------
python -m pip install onnxruntime-gpu==1.13.1
echo Testing...
python -c "import onnxruntime as ort; print(f'ONNX Runtime v{ort.__version__} providers: {ort.get_available_providers()}')" || goto :try_112

echo Success with version 1.13.1! Try running your application now.
goto :end

:try_112
echo ------------- Trying v1.12.0 -------------
python -m pip uninstall onnxruntime onnxruntime-gpu -y
python -m pip install onnxruntime-gpu==1.12.0
echo Testing...
python -c "import onnxruntime as ort; print(f'ONNX Runtime v{ort.__version__} providers: {ort.get_available_providers()}')" || goto :try_111

echo Success with version 1.12.0! Try running your application now.
goto :end

:try_111
echo ------------- Trying v1.11.0 -------------
python -m pip uninstall onnxruntime onnxruntime-gpu -y
python -m pip install onnxruntime-gpu==1.11.0
echo Testing...
python -c "import onnxruntime as ort; print(f'ONNX Runtime v{ort.__version__} providers: {ort.get_available_providers()}')" || goto :try_110

echo Success with version 1.11.0! Try running your application now.
goto :end

:try_110
echo ------------- Trying v1.10.0 -------------
python -m pip uninstall onnxruntime onnxruntime-gpu -y
python -m pip install onnxruntime-gpu==1.10.0
echo Testing...
python -c "import onnxruntime as ort; print(f'ONNX Runtime v{ort.__version__} providers: {ort.get_available_providers()}')" || goto :try_cpu

echo Success with version 1.10.0! Try running your application now.
goto :end

:try_cpu
echo ------------- All GPU versions failed, trying CPU as last resort -------------
python -m pip uninstall onnxruntime onnxruntime-gpu -y
python -m pip install onnxruntime==1.15.1
echo Testing CPU version...
python -c "import onnxruntime as ort; print(f'ONNX Runtime v{ort.__version__} providers: {ort.get_available_providers()}')" || goto :failed

echo CPU version works. This will allow your application to run without GPU acceleration for ONNX operations.
goto :end

:failed
echo All versions failed. Please check your CUDA installation and system configuration.
echo You may need to install the appropriate Visual C++ Redistributable packages.

:end
pause 