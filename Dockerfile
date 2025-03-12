FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    espeak \
    ffmpeg \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install pip dependencies separately for better caching
COPY requirements-fixed.txt /app/
COPY requirements.txt /app/
COPY requirements_extra.txt /app/
COPY wheels/ /app/wheels/

# Install build tools with specific versions to avoid conflicts
RUN pip install --no-cache-dir --upgrade pip==24.0 setuptools==69.0.3 wheel ninja

# Install critical base dependencies first
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.15.0 protobuf==3.19.6

# Install CUDA-specific packages
RUN pip install --no-cache-dir -r requirements_extra.txt

# Install dependencies with fixed versions
RUN pip install --no-cache-dir -r requirements-fixed.txt

# Ensure protobuf version is correct after requirements are installed
RUN pip install --no-cache-dir --force-reinstall protobuf==3.19.6

# Install remaining packages
RUN pip install --no-cache-dir \
    mamba-ssm[causal-conv1d] --no-build-isolation \
    TTS \
    fairseq \
    whisperx

# Install any remaining dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Fix any dependency conflicts by reinstalling critical packages
RUN pip install --no-cache-dir --force-reinstall numpy==1.26.4 scipy==1.15.0 protobuf==3.19.6 setuptools==69.0.3

# Install ONNX Runtime with careful version management
RUN pip uninstall -y onnxruntime onnxruntime-gpu && \
    pip install --no-cache-dir protobuf==3.19.6 --force-reinstall && \
    pip install --no-cache-dir onnxruntime==1.16.3 && \
    python -c "import onnxruntime; print('ONNX Runtime CPU version: ' + onnxruntime.__version__)" && \
    pip install --no-cache-dir onnxruntime-gpu==1.16.3 || \
    (echo "Trying older version of ONNX Runtime GPU..." && \
     pip install --no-cache-dir onnxruntime-gpu==1.15.1 || \
     echo "Using CPU version instead")

# Install custom wheel
RUN pip install --no-cache-dir ./wheels/audiosr-0.0.8-py2.py3-none-any.whl

# Clone and install Python-Wrapper-for-World-Vocoder
RUN git clone https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder.git \
    && cd Python-Wrapper-for-World-Vocoder \
    && git submodule update --init \
    && pip install --no-cache-dir . \
    && cd .. \
    && rm -rf Python-Wrapper-for-World-Vocoder

# Copy application code
COPY . /app/

# Expose ports
EXPOSE 7861

# Start command
CMD ["python", "main.py", "--listen", "--port", "7861"] 