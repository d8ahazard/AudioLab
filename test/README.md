# AudioLab Testing Framework

This directory contains scripts and utilities for testing the various components of AudioLab.

## Overview

The AudioLab testing framework consists of two main components:

1. **Module Testing** - Test individual layout modules like diffrythm, music, orpheus, etc.
2. **Wrapper Testing** - Test individual audio processing wrappers like compare, convert, merge, etc.

## Test Scripts

### `test_modules.py`

This script allows testing of layout modules. It:

- Automatically discovers all layout modules in the `/layouts` directory
- Presents a menu to select which module(s) to test
- Looks for a `test()` function in each module
- If no dedicated test function is found, it lists available functions that could be used for testing

Usage:
```bash
python -m test.test_modules
```

### `test_wrappers.py`

This script allows testing of audio processing wrappers. It:

- Automatically discovers all wrapper modules in the `/wrappers` directory
- Presents a menu to select which wrapper(s) to test
- Looks for a `test()` method in each wrapper class
- If no dedicated test method is found, it displays information about the wrapper

Usage:
```bash
python -m test.test_wrappers
```

## Adding Tests to Components

### Adding Tests to Layout Modules

To make a layout module testable, add a `test()` function:

```python
def test():
    """
    Test function for the module.
    
    This function should test the core functionality of the module
    without requiring external dependencies where possible.
    """
    print("Running test...")
    
    # Test implementation here
    
    print("Test completed successfully!")
    return True
```

### Adding Tests to Wrappers

To make a wrapper testable, add a `test()` method to the wrapper class:

```python
def test(self):
    """
    Test method for the wrapper.
    
    This method should test the core functionality of the wrapper
    and should be self-contained.
    """
    print(f"Running {self.title} wrapper test...")
    
    # Test implementation here
    
    print(f"{self.title} wrapper test completed successfully!")
```

## Best Practices for Writing Tests

1. **Self-Contained**: Tests should create any necessary data and clean up after themselves.
2. **Minimal Dependencies**: Try to minimize external dependencies where possible.
3. **Temporary Resources**: Use `tempfile.TemporaryDirectory()` for creating temporary test files.
4. **Proper Cleanup**: Always clean up temporary files and resources in a `finally` block.
5. **Clear Output**: Use clear print statements to indicate progress and success/failure.
6. **Error Handling**: Use proper exception handling to catch and report errors.

## Example Test

Here's an example of a good test implementation:

```python
def test(self):
    print(f"Running {self.title} test...")
    
    # Create temporary directory
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp(prefix="audiolab_test_")
    
    try:
        # Create test data
        # ...
        
        # Run the actual test
        # ...
        
        # Verify results
        # ...
        
        print("Test passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
```

## Future Improvements

- Add automated CI/CD integration
- Add coverage reporting
- Add benchmark tests for performance-critical components 