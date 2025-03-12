# AudioLab Dependency Management

This document outlines the different approaches available for managing the complex dependencies required by AudioLab.

## The Challenge

AudioLab requires numerous packages that often have conflicting dependencies, particularly around:
- Different versions of numpy required by different packages
- CUDA compatibility requirements
- PyTorch ecosystem dependencies
- Specialized audio processing libraries
- Conflicting protobuf version requirements (e.g., descript-audiotools vs onnx)
- Setuptools and numpy compatibility issues
- ONNX Runtime DLL loading issues on Windows

## Available Solutions

### 1. Structured Installation Scripts (setup.bat / setup.sh)

The setup scripts have been reorganized to install dependencies in a specific order using dependency groups:

1. Critical base dependencies (numpy, scipy, protobuf, setuptools) installed first with fixed versions
2. PyTorch ecosystem packages installed with proper CUDA support
3. CUDA-specific packages from requirements_extra.txt
4. Main dependencies with fixed versions from requirements-fixed.txt
5. Special installations (World Vocoder)
6. Wheel installations
7. TTS and remaining packages from requirements.txt
8. Final fixes for potential conflicts
9. ONNX Runtime with specific versions and fallbacks

This approach helps minimize conflicts by controlling the order of installation and reinstalling critical packages at the end. The use of requirements-fixed.txt ensures all dependencies are installed with consistent, pinned versions.

### 2. Conda Environment (environment.yml)

For more robust dependency management, you can use Conda:

```bash
# Install Miniconda if not already installed
# Create and activate the environment
conda env create -f environment.yml
conda activate audiolab
```

Conda provides better binary dependency handling and conflict resolution than pip.

### 3. Docker Container (Dockerfile)

The most isolated and reproducible approach is using Docker:

```bash
# Build the Docker image
docker build -t audiolab .

# Run the container
docker run -p 7861:7861 --gpus all audiolab
```

Benefits of using Docker:
- Complete isolation from system dependencies
- Reproducible environment across different machines
- Easier deployment and sharing

### 4. Fixed Dependencies (requirements-fixed.txt)

We've created requirements-fixed.txt with pinned versions of all dependencies to avoid version conflicts. This file is used in all of our setup methods (scripts, conda, docker) to ensure consistency.

## Dependency Installation Order

The correct installation order is crucial for resolving conflicts:

1. Install foundational packages with exact versions (numpy, scipy, protobuf, setuptools) first
2. Install PyTorch ecosystem packages with CUDA support
3. Install CUDA-specific packages from requirements_extra.txt
4. Install main dependencies with fixed versions from requirements-fixed.txt
5. Install specialized packages and wheels
6. Install any remaining packages from the original requirements.txt
7. Fix potential conflicts by reinstalling critical packages
8. Handle ONNX Runtime installation with careful version management

## Handling Common Dependency Conflicts

### ONNX Runtime Issues on Windows

ONNX Runtime frequently has issues on Windows due to several factors:
- DLL loading errors with the GPU version
- Visual C++ Redistributable dependencies
- CUDA version compatibility problems
- Protobuf version conflicts

Our solution uses a tiered approach:
1. Install Visual C++ Redistributables before Python packages
2. Install the CPU version of ONNX Runtime first and test it works
3. Try multiple versions of ONNX Runtime if the default version fails
4. Provide fallback to CPU-only version if GPU version fails
5. Give clear recovery instructions for users

### Setuptools and Numpy Compatibility

A critical issue exists between newer setuptools versions and numpy:
- Newer setuptools (>=70.0.0) reorganizes distutils imports
- Numpy uses distutils for some components (particularly build-related ones)
- This can result in `ImportError: cannot import name 'compiler_class' from 'distutils.ccompiler'`

Our solution:
1. Pin setuptools version to 69.0.3 which is compatible with numpy 1.26.4
2. Install setuptools early in the process
3. Reinstall setuptools after all dependencies to ensure it's not overridden

### Protobuf Version Conflicts

One major conflict in AudioLab involves protobuf versions:
- descript-audiotools requires protobuf<3.20
- onnx requires protobuf>=3.20.2

Our solution uses protobuf==3.19.6 as a compromise version and forces its installation:
1. Before installing any other packages
2. After installing major dependency groups
3. Specifically before installing onnxruntime-gpu

### Numpy/Scipy Conflicts

These are common Python package conflicts that we address by:
1. Installing specific versions (numpy==1.26.4, scipy==1.15.0) early in the process
2. Reinstalling these packages with --force-reinstall after all other packages

## Best Practices for Adding New Dependencies

When adding new dependencies to AudioLab:

1. **Test in isolation first**: Install the new package in a fresh virtual environment to identify its dependencies
2. **Pin the version**: Always specify exact versions for all packages
3. **Check for conflicts**: Run `pip check` to detect dependency conflicts
4. **Update requirements-fixed.txt**: Add the new package with its pinned version
5. **Group logically**: Add the package to the appropriate group in the setup scripts
6. **Verify compatibility**: Ensure it works with our pinned versions of critical packages (numpy, scipy, protobuf, setuptools)

## Emergency Fixes

If you encounter dependency issues when running AudioLab, try:

```bash
# For numpy/scipy/protobuf/setuptools conflicts (most common)
pip install numpy==1.26.4 scipy==1.15.0 protobuf==3.19.6 setuptools==69.0.3 --force-reinstall

# For ONNX Runtime issues on Windows
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime==1.15.1

# For PyTorch conflicts
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --force-reinstall

# If you encounter "ImportError: cannot import name" errors
# Try reinstalling the specific package with the import error
```

## Windows-Specific Dependencies

On Windows, these additional steps may be needed:
1. Install Visual C++ Redistributable packages (the setup script does this automatically)
2. Make sure your CUDA installation matches the version expected by onnxruntime-gpu
3. Try using CPU-only versions of packages if GPU versions fail

## Dependency Isolation Strategies

For particularly troublesome dependencies, consider:

1. **Separate environments**: Use multiple virtual environments when packages can't coexist
2. **Container-based approach**: Isolate problematic packages in containers
3. **Use --no-deps flag**: For packages with problematic dependencies that you've already satisfied

## Future Improvements

Consider implementing:
1. **Dependency locking**: Use pip-tools or Poetry to generate lockfiles
2. **Modular dependencies**: Split requirements by function (core, audio, ml, etc.)
3. **CI testing**: Set up CI to test dependency installations on different platforms
4. **Dependency health monitoring**: Regular checks for security updates and compatibility improvements 