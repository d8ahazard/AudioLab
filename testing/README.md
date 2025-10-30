# AudioLab Comprehensive Testing Framework

## Overview

This document outlines a comprehensive testing strategy for AudioLab, a complex AI audio processing application with multiple modules for voice conversion, text-to-speech, music generation, audio separation, and more. The framework ensures 100% test coverage across all components while maintaining modularity and ease of use.

## Architecture

The testing framework is organized into three main layers:

1. **Unit Tests** - Individual function/component testing
2. **Integration Tests** - Cross-module interaction testing
3. **System Tests** - End-to-end workflow testing

## Directory Structure

```
testing/
├── README.md                    # This file
├── requirements.txt             # Testing dependencies
├── config/                      # Test configuration files
│   ├── test_settings.json       # Global test settings
│   ├── audio_samples/           # Test audio files
│   └── model_configs/           # Test model configurations
├── unit/                        # Unit tests
│   ├── layouts/                 # Layout module tests
│   ├── modules/                 # Core module tests
│   ├── handlers/                # Handler tests
│   └── wrappers/                # Wrapper tests
├── integration/                 # Integration tests
│   ├── workflows/               # End-to-end workflow tests
│   └── cross_module/            # Cross-module interaction tests
├── system/                      # System-level tests
│   └── e2e/                     # End-to-end scenario tests
├── utils/                       # Testing utilities
│   ├── test_helpers.py          # Common test functions
│   ├── audio_validation.py      # Audio file validation utilities
│   ├── model_mocking.py         # Model mocking utilities
│   └── test_data_generator.py   # Test data generation
└── reports/                     # Test reports and metrics
    ├── coverage/                # Coverage reports
    └── performance/             # Performance benchmarks
```

## Test Categories by Component

### 1. Layout Modules (`layouts/`)

#### `rvc_train.py` - RVC Training Layout
**Core Functions to Test:**
- `preprocess_dataset()` - Dataset preprocessing
- `extract_f0_feature()` - F0 feature extraction
- `click_train()` - Model training process
- `train_index()` - Index building
- `validate_model_and_index()` - Model validation

**Test Scenarios:**
- Dataset preprocessing with various audio formats
- F0 extraction with different algorithms (RMVPE, DIO, etc.)
- Training workflow with mocked data
- Index building and validation
- Model validation with corrupted/invalid models

#### `music.py` - Music Generation Layout
**Core Functions to Test:**
- `generate_callback()` - Music generation process
- `generate_yue_music()` - YuE model inference
- `update_model_selection()` - Model selection logic
- `download_output_files()` - Output file management

**Test Scenarios:**
- Music generation with different prompts and styles
- Model selection and switching
- Output file creation and cleanup
- Progress tracking and error handling

#### `process.py` - Audio Processing Pipeline
**Core Functions to Test:**
- `process()` - Main processing pipeline
- `get_processor()` - Processor instantiation
- `check_processor_conflicts()` - Conflict detection
- `update_preview()` - Preview generation
- `list_projects()` - Project management

**Test Scenarios:**
- Processing pipeline with multiple processors
- Processor conflict detection and resolution
- Preview generation for various file types
- Project loading and management
- Error handling in processing chain

#### `tts.py` - Text-to-Speech Layout
**Core Functions to Test:**
- `run_zonos_tts()` - Zonos TTS inference
- `run_chatterbox_tts()` - Chatterbox TTS inference
- `run_dia_tts()` - DIA TTS inference
- `download_model()` - Model downloading
- `set_espeak_lib_path_win()` - Platform-specific setup

**Test Scenarios:**
- TTS generation with different models and voices
- Model downloading and caching
- Platform-specific initialization
- Multi-language support testing
- Emotion and style parameter testing

#### `stable_audio.py` - Stable Audio Layout
**Core Functions to Test:**
- Audio generation with prompts
- Model inference and output validation
- Parameter validation and error handling

#### `acestep.py` - ACE-Step Music Generation
**Core Functions to Test:**
- ACE-Step model inference
- Audio-to-audio conversion
- Music generation workflows

#### `align.py` - Audio Alignment
**Core Functions to Test:**
- Audio alignment algorithms
- Synchronization processes
- Output validation

#### `transcribe.py` - Audio Transcription
**Core Functions to Test:**
- Transcription accuracy
- Multi-language support
- Timestamp generation

#### `wavetransfer.py` - Wave Transfer Processing
**Core Functions to Test:**
- Wave transfer algorithms
- Audio transformation processes

### 2. Core Modules (`modules/`)

#### `rvc/` - Retrieval-based Voice Conversion
**Components to Test:**
- `infer/modules/train/` - Training pipeline
- `infer/modules/vc/` - Voice conversion
- `infer/modules/preprocess/` - Preprocessing
- `lib/models.py` - Model architectures
- `utils.py` - Utility functions
- `pitch_extraction.py` - Pitch extraction algorithms

**Test Scenarios:**
- Model training with synthetic datasets
- Voice conversion with different speakers
- Pitch extraction accuracy testing
- Model inference performance
- Configuration validation

#### `zonos/` - Zonos TTS Module
**Components to Test:**
- `model.py` - Core model implementation
- `conditioning.py` - Input conditioning
- `speaker_cloning.py` - Speaker embedding
- `autoencoder.py` - Audio encoding/decoding

**Test Scenarios:**
- Speaker embedding generation
- Audio encoding/decoding
- Conditioning parameter validation
- Model inference with various inputs

#### `yue/` - YuE Music Generation
**Components to Test:**
- `inference/infer.py` - Main inference
- `inference/codecmanipulator.py` - Codec handling
- `inference/mmtokenizer.py` - Tokenization

**Test Scenarios:**
- Music generation with various prompts
- Codec manipulation testing
- Tokenization accuracy

#### `diatts/` - DIA TTS Module
**Components to Test:**
- `dia/` - Core DIA implementation
- Model loading and inference
- Audio processing pipeline

#### `cloning/` - Voice Cloning
**Components to Test:**
- `main.py` - Main cloning logic
- `openvoice.py` - OpenVoice integration
- `tts.py` - TTS functionality

**Test Scenarios:**
- Voice cloning from samples
- Speaker separation testing
- TTS quality validation

#### `separator/` - Audio Separation
**Components to Test:**
- `stem_separator.py` - Stem separation logic
- Separation quality metrics
- Multi-instrument handling

#### `stable_audio/` - Stable Audio Model
**Components to Test:**
- `model.py` - Model implementation
- Audio generation quality
- Parameter validation

#### `voicecraft/` - VoiceCraft Module
**Components to Test:**
- Model inference
- Audio processing
- Integration testing

#### `wavetransfer/` - Wave Transfer
**Components to Test:**
- `main.py` - Main processing
- `model.py` - Model architecture
- Diffusion processes

#### `acestep/` - ACE-Step Module
**Components to Test:**
- Model inference
- Audio processing
- Generation quality

#### `rtla/` - Real-Time Audio Processing
**Components to Test:**
- `stream_processor.py` - Stream processing
- Real-time performance
- Buffer management

### 3. Handlers (`handlers/`)

#### Core Handler Functions
- `args.py` - Argument parsing and validation
- `config.py` - Configuration management
- `download.py` - Model/file downloading
- `processing.py` - Processing pipeline management
- `tts.py` - TTS-specific handling

**Test Scenarios:**
- Configuration loading and validation
- Argument parsing edge cases
- Download error handling
- Processing pipeline coordination

### 4. Wrappers (`wrappers/`)

#### Wrapper Classes
- `base_wrapper.py` - Base wrapper functionality
- `clone.py` - Voice cloning operations
- `compare.py` - Audio comparison tools
- `convert.py` - Format conversion
- `export.py` - Export functionality
- `merge.py` - Audio merging
- `remaster.py` - Audio remastering
- `separate.py` - Audio separation
- `super_res.py` - Super resolution

**Test Scenarios:**
- Wrapper initialization and configuration
- Processing pipeline execution
- Error handling and recovery
- Output validation

## Testing Implementation Strategy

### 1. Test Data Management

#### Audio Test Files
- **Clean speech samples** (various speakers, languages, accents)
- **Music samples** (different genres, instruments, quality levels)
- **Noise samples** (background noise, artifacts, distortions)
- **Mixed audio** (speech + music, multi-speaker conversations)

#### Model Test Configurations
- **Minimal models** for fast testing
- **Corrupted models** for error testing
- **Edge case configurations** for boundary testing

### 2. Mocking Strategy

#### Model Mocking
- Create lightweight model mocks for fast testing
- Mock external API calls (HuggingFace, etc.)
- Mock file I/O operations

#### Environment Mocking
- Mock GPU availability
- Mock file system operations
- Mock network connectivity

### 3. Test Utilities

#### Audio Validation
```python
def validate_audio_quality(audio_path, expected_duration=None, expected_sample_rate=None):
    """Validate audio file properties and quality metrics"""

def compare_audio_files(original, processed, tolerance=0.1):
    """Compare two audio files for similarity"""

def measure_audio_metrics(audio_path, metrics=['snr', 'rmse', 'pesq']):
    """Calculate audio quality metrics"""
```

#### Model Testing Helpers
```python
def create_test_model_config(model_type, parameters):
    """Create test model configuration"""

def mock_model_inference(input_data, model_type):
    """Mock model inference for testing"""

def validate_model_output(output, expected_format):
    """Validate model output format and structure"""
```

### 4. Test Organization

#### Test File Naming Convention
```
test_[component]_[functionality]_[scenario].py
```

Examples:
- `test_rvc_train_preprocessing.py`
- `test_tts_zonos_inference.py`
- `test_music_generation_prompts.py`
- `test_process_pipeline_integration.py`

#### Test Class Structure
```python
class TestRVCTraining(unittest.TestCase):
    def setUp(self):
        # Test setup
        pass

    def tearDown(self):
        # Cleanup
        pass

    def test_preprocess_dataset_valid_input(self):
        # Test case
        pass

    def test_extract_f0_features_edge_cases(self):
        # Test case
        pass
```

## Implementation Plan

### Phase 1: Foundation (Week 1-2)
1. Create directory structure and base test files
2. Implement test utilities and helpers
3. Set up test configuration management
4. Create audio validation utilities
5. Implement model mocking framework

### Phase 2: Unit Tests (Week 3-6)
1. **Week 3**: Layout module unit tests (rvc_train, music, process, tts)
2. **Week 4**: Core module unit tests (RVC, Zonos, YuE, DIA)
3. **Week 5**: Handler and wrapper unit tests
4. **Week 6**: Integration between related components

### Phase 3: Integration Tests (Week 7-8)
1. **Week 7**: Cross-module workflows
2. **Week 8**: End-to-end pipeline testing

### Phase 4: System Tests (Week 9-10)
1. **Week 9**: Full application workflows
2. **Week 10**: Performance and load testing

## Usage Instructions

### Running Tests

#### Run All Tests
```bash
cd testing
python -m pytest --cov=audiolab --cov-report=html
```

#### Run Specific Test Category
```bash
# Unit tests only
python -m pytest unit/ -v

# Integration tests only
python -m pytest integration/ -v

# Specific component tests
python -m pytest unit/modules/test_rvc.py -v
```

#### Run Tests with Coverage
```bash
python -m pytest --cov=audiolab --cov-report=xml --cov-report=html
```

### Adding New Tests

1. **Identify the component** to test
2. **Create test file** following naming convention
3. **Implement test class** inheriting from appropriate base class
4. **Add test methods** for each scenario
5. **Include proper setup/teardown**
6. **Add test data** if needed
7. **Update this documentation**

### Test Data Requirements

#### Minimum Test Data Set
- 10 diverse speech samples (5 male, 5 female, multiple languages)
- 5 music samples (different genres)
- 3 noise samples (different types)
- 2 model configurations per major component

#### Test Data Organization
```
testing/config/audio_samples/
├── speech/
│   ├── english_male_001.wav
│   ├── english_female_001.wav
│   └── multilingual_samples/
├── music/
│   ├── classical_001.wav
│   ├── rock_001.wav
│   └── electronic_001.wav
└── noise/
    ├── background_001.wav
    └── artifacts_001.wav
```

## Quality Metrics

### Coverage Targets
- **Unit Tests**: 90%+ coverage
- **Integration Tests**: 80%+ coverage
- **Overall**: 85%+ coverage

### Performance Benchmarks
- **Test Execution Time**: < 5 minutes for full suite
- **Memory Usage**: < 2GB peak usage
- **Individual Test Timeout**: < 30 seconds

### Quality Gates
- All tests must pass before deployment
- Coverage must meet minimum thresholds
- Performance benchmarks must be maintained
- No critical security vulnerabilities

## Maintenance and Evolution

### Regular Updates
- Review and update tests with code changes
- Add tests for new features immediately
- Remove obsolete tests regularly
- Update test data as needed

### Continuous Improvement
- Monitor test execution metrics
- Identify and fix flaky tests
- Optimize slow-running tests
- Enhance test coverage in under-tested areas

## Troubleshooting

### Common Issues
- **Model Download Failures**: Check network connectivity and credentials
- **GPU Memory Issues**: Reduce batch sizes in tests
- **File Path Issues**: Use absolute paths in test configurations
- **Dependency Conflicts**: Isolate test environments

### Debug Mode
```bash
python -m pytest -v -s --pdb  # Post-mortem debugging
python -m pytest -v --durations=10  # Show slowest tests
python -m pytest -v --tb=short  # Shorter traceback format
```

## Contributing

When adding new tests:
1. Follow the established patterns
2. Include comprehensive documentation
3. Add appropriate test data
4. Update this README
5. Ensure CI/CD compatibility

For questions or issues, contact the development team or create an issue in the project repository.
