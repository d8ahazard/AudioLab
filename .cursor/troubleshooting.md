# AudioLab Troubleshooting Guide

This document provides common troubleshooting approaches for the AudioLab project to help AI agents diagnose and fix issues effectively.

## Common Issue Categories

### UI-Related Issues

1. **Broken UI Elements**
   - Check layouts/*.py files for the affected tab
   - Verify element IDs match between UI components and event listeners
   - Check for typos in element IDs or classes
   - Ensure all UI elements have proper registration in register_descriptions()

2. **Event Handling Problems**
   - Check the listen() function in the relevant layout file
   - Verify event handlers are properly registered
   - Check for mismatches between input/output elements and handler functions
   - Verify global variables are properly initialized

3. **Tab Communication Issues**
   - Check ArgHandler usage for cross-tab element retrieval
   - Verify element IDs are consistent between tabs
   - Check for null/None element references when accessing other tabs
   - Ensure the "Send to Process" button is properly connected

### Processing Issues

1. **Wrapper Processing Failures**
   - Check the process_audio() method in the wrapper
   - Verify input validation in validate_args()
   - Check for proper error handling around external libraries
   - Look for file path/permissions issues with inputs/outputs

2. **Audio Processing Problems**
   - Check the AudioTrack handling in the affected wrapper
   - Verify sample rate and channel consistency
   - Check for memory issues with large audio files
   - Ensure temporary files are properly cleaned up

3. **Model Loading Issues**
   - Check for missing model files in the expected directory
   - Verify model versions match the code requirements
   - Check for CUDA compatibility issues
   - Ensure sufficient VRAM is available for the model

### API Issues

1. **Endpoint Registration Problems**
   - Check register_api_endpoint() implementation in the affected module
   - Verify endpoint URL patterns follow conventions
   - Check parameter validation in the endpoint functions
   - Ensure proper error handling and responses

2. **Request Processing Issues**
   - Check request parsing and validation
   - Verify file upload handling for audio files
   - Check for proper content type and response formatting
   - Ensure asynchronous tasks have proper status reporting

## Module-Specific Troubleshooting

### Process Tab
- Check wrapper discovery and initialization in layouts/process.py
- Verify processing chain execution in process.py
- Check for priority conflicts between wrappers
- Ensure ProjectFiles is properly managing the file hierarchy

### DiffRhythm
- Check model loading in modules/diffrythm/
- Verify chunked decoding implementation for memory optimization
- Check LRC parsing for lyric-based generation
- Ensure proper audio format handling for style references

### WaveTransfer
- Check project management system in modules/wavetransfer/
- Verify training/inference separation in the code
- Check chunked processing for longer audio files
- Ensure proper model checkpointing during training

### Orpheus TTS
- Check voice model loading in modules/orpheus/
- Verify text preprocessing for speech generation
- Check emotion/style parameter handling
- Ensure proper audio output format and quality

### Stable Audio
- Check model initialization in modules/stable_audio/
- Verify proper text prompt processing
- Check guidance scale and other generation parameters
- Ensure output duration handling is correct

### Audio Separation
- Check model selection logic in wrappers/separate.py
- Verify stem selection and processing
- Check for GPU memory issues with large files
- Ensure proper output naming and organization

## Performance Troubleshooting

1. **Memory Usage Issues**
   - Check for memory leaks in audio processing loops
   - Verify large tensors are properly released after use
   - Check for unnecessary copies of audio data
   - Consider implementing chunked processing for large files

2. **Processing Speed Issues**
   - Check for bottlenecks in CPU-bound operations
   - Verify GPU utilization for model inference
   - Check for unnecessary conversions between audio formats
   - Consider batch processing when appropriate

3. **GPU Utilization Problems**
   - Check for proper CUDA device selection
   - Verify models are loaded on the correct device
   - Check for unnecessary CPU-GPU transfers
   - Ensure appropriate batch sizes for available VRAM

## Common Code Patterns to Check

1. **Error Handling Gaps**
   - Look for bare except blocks
   - Check for proper error propagation up the call stack
   - Verify informative error messages are returned to users
   - Ensure cleanup happens even when errors occur

2. **Resource Management Issues**
   - Check for proper file handle closing
   - Verify temporary files are cleaned up
   - Check for GPU memory management issues
   - Ensure background processes are properly terminated

3. **Path Handling Problems**
   - Check for hardcoded paths
   - Verify path joins use os.path.join()
   - Check for proper handling of relative paths
   - Ensure consistent path normalization 