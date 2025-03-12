# ONNX Runtime Troubleshooting Guide for AudioLab

This guide provides solutions for common ONNX Runtime GPU issues on Windows systems.

## Quick Fix Scripts

We've provided several scripts to help diagnose and fix ONNX Runtime issues:

1. **`fix_onnx.bat`** - Primary fix script that:
   - Sets required environment variables
   - Configures CUDA paths
   - Reinstalls critical dependencies
   - Installs ONNX Runtime GPU v1.14.1 (often works well on Windows)
   - Tests if GPU acceleration is available

2. **`fix_onnx_fallback.bat`** - Tries multiple ONNX Runtime versions if the primary fix fails:
   - Automatically tries versions 1.13.1, 1.12.0, 1.11.0, and 1.10.0
   - Falls back to CPU version only as a last resort
   - Reports which version worked

3. **`check_system.bat`** - Diagnostic tool that:
   - Checks Python environment
   - Verifies CUDA installation
   - Examines PyTorch CUDA detection
   - Inspects environment variables
   - Tests ONNX Runtime providers
   - Looks for DLL issues
   - Checks Visual C++ Redistributables
   - Tests DirectML availability

## Common Issues and Solutions

### DLL Loading Error

If you see errors like:
```
ImportError: DLL load failed while importing onnxruntime_pybind11_state: A dynamic link library (DLL) initialization routine failed.
```

Try these steps in order:

1. Run `check_system.bat` to diagnose the issue
2. Run `fix_onnx.bat` to apply the primary fix
3. If that fails, run `fix_onnx_fallback.bat` to try alternative versions

### Root Causes

These errors are typically caused by:

1. **CUDA/CUDNN version mismatch** with your installed ONNX Runtime
2. **Missing Visual C++ Redistributables** (install latest from Microsoft)
3. **Incorrect environment variables** (our fix scripts set these correctly)
4. **Path issues** with CUDA binaries (our scripts add these to PATH)

### For Advanced Users

If the scripts don't resolve your issue:

1. Check your GPU compatibility with CUDA using `nvidia-smi`
2. Install Visual C++ Redistributables directly from Microsoft
3. Try a different CUDA version (if possible)
4. Consider using DirectML provider instead of CUDA on Windows
5. Look for specific version compatibility with your NVIDIA drivers

## Reporting Unresolved Issues

If you've tried all the scripts and still have issues:

1. Run `check_system.bat`
2. Capture all the output
3. Note which specific error messages you're seeing
4. Report the issue along with the diagnostic information

## Emergency Fallback

If you need to get the application running immediately without GPU acceleration:

```
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime==1.15.1
```

This will enable CPU-only mode until the GPU issues can be resolved. 