import os
import sys
import importlib.util
import inspect
from typing import Dict, List, Callable, Any, Optional

# Add the parent directory to sys.path so we can import from wrappers
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


def discover_wrapper_modules() -> Dict[str, str]:
    """
    Discover all wrapper modules in the wrappers directory.
    
    Returns:
        Dict[str, str]: Dictionary mapping module names to their file paths
    """
    wrappers_dir = os.path.join(parent_dir, "wrappers")
    wrapper_files = {}
    
    if not os.path.exists(wrappers_dir):
        print(f"Error: Wrappers directory not found at {wrappers_dir}")
        return {}
    
    for filename in os.listdir(wrappers_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            # Skip base_wrapper.py when testing as it's abstract
            if filename == "base_wrapper.py":
                continue
                
            module_name = filename[:-3]  # Remove the .py extension
            file_path = os.path.join(wrappers_dir, filename)
            wrapper_files[module_name] = file_path
    
    return wrapper_files


def load_module(module_path: str, module_name: str) -> Optional[Any]:
    """
    Load a Python module from its file path.
    
    Args:
        module_path (str): Path to the module file
        module_name (str): Name to give the module
        
    Returns:
        Optional[Any]: The loaded module or None if loading failed
    """
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            print(f"Error: Could not load spec for {module_name} from {module_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error loading module {module_name}: {e}")
        return None


def find_wrapper_class(module: Any) -> Optional[Any]:
    """
    Find the wrapper class in a module.
    
    Args:
        module (Any): The loaded module
        
    Returns:
        Optional[Any]: The wrapper class or None if not found
    """
    from wrappers.base_wrapper import BaseWrapper
    
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and 
            issubclass(obj, BaseWrapper) and 
            obj.__module__ == module.__name__):
            return obj
    return None


def test_wrapper(module_path: str, module_name: str) -> bool:
    """
    Test a specific wrapper module.
    
    Args:
        module_path (str): Path to the module file
        module_name (str): Name of the module
        
    Returns:
        bool: True if test was successful, False otherwise
    """
    print(f"\nTesting wrapper: {module_name}")
    print("=" * 50)
    
    # Load the module
    module = load_module(module_path, module_name)
    if not module:
        return False
    
    # Find the wrapper class
    wrapper_class = find_wrapper_class(module)
    if not wrapper_class:
        print(f"No wrapper class found in {module_name}")
        return False
    
    print(f"Found wrapper class: {wrapper_class.__name__}")
    
    # Check if there's a test method
    if hasattr(wrapper_class, "test"):
        try:
            print(f"Calling test method on {wrapper_class.__name__}")
            # Try to create an instance and call test
            wrapper_instance = wrapper_class()
            wrapper_instance.test()
            print(f"✅ Test for {module_name} completed successfully")
            return True
        except Exception as e:
            print(f"❌ Error testing {module_name}: {e}")
            return False
    
    # Otherwise, just report on the methods available
    print(f"No test method found in {wrapper_class.__name__}")
    print("The following methods could potentially be used for testing:")
    
    for name, method in inspect.getmembers(wrapper_class, predicate=inspect.isfunction):
        if not name.startswith("_"):  # Skip private methods
            print(f"  - {name}")
    
    # Get information about the wrapper
    print("\nWrapper information:")
    try:
        instance = wrapper_class()
        print(f"  Title: {instance.title}")
        print(f"  Description: {instance.description}")
        print(f"  Priority: {instance.priority}")
        
        if hasattr(instance, "allowed_kwargs") and instance.allowed_kwargs:
            print("\n  Available settings:")
            for key, value in instance.allowed_kwargs.items():
                print(f"    - {key}: {value.default} ({value.description})")
    except Exception as e:
        print(f"  Error instantiating wrapper: {e}")
    
    print(f"⚠️ Manual testing required for {module_name}")
    return False


def main():
    """Main function to discover and test wrapper modules."""
    wrapper_files = discover_wrapper_modules()
    
    if not wrapper_files:
        print("No wrapper modules found!")
        return
    
    # Create a sorted list of modules
    module_names = sorted(wrapper_files.keys())
    
    print("AudioLab Wrapper Module Tester")
    print("=" * 50)
    print("Available wrappers:")
    print("1. All wrappers")
    
    for i, name in enumerate(module_names, 2):
        print(f"{i}. {name}")
    
    choice = input("\nEnter number of wrapper to test (or 'q' to quit): ")
    
    if choice.lower() == 'q':
        return
    
    try:
        choice = int(choice)
        if choice == 1:
            # Test all wrappers
            print("\nTesting all wrappers...")
            for name, path in wrapper_files.items():
                test_wrapper(path, name)
        elif 2 <= choice <= len(module_names) + 1:
            # Test specific wrapper
            module_index = choice - 2
            module_name = module_names[module_index]
            module_path = wrapper_files[module_name]
            test_wrapper(module_path, module_name)
        else:
            print("Invalid selection.")
    except ValueError:
        print("Please enter a valid number.")


if __name__ == "__main__":
    main() 