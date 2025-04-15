import os
import sys
import importlib.util
import inspect
from typing import Dict, List, Callable, Any, Optional


def discover_layout_modules() -> Dict[str, str]:
    """
    Discover all layout modules in the layouts directory.
    
    Returns:
        Dict[str, str]: Dictionary mapping module names to their file paths
    """
    layouts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "layouts")
    layout_files = {}
    
    if not os.path.exists(layouts_dir):
        print(f"Error: Layouts directory not found at {layouts_dir}")
        return {}
    
    for filename in os.listdir(layouts_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]  # Remove the .py extension
            file_path = os.path.join(layouts_dir, filename)
            layout_files[module_name] = file_path
    
    return layout_files


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


def find_testable_functions(module: Any) -> Dict[str, Callable]:
    """
    Find all functions in a module that could be used for testing.
    
    Args:
        module (Any): The loaded module
        
    Returns:
        Dict[str, Callable]: Dictionary mapping function names to functions
    """
    test_functions = {}
    
    # Look for explicit test methods first
    if hasattr(module, "test"):
        test_functions["test"] = getattr(module, "test")
    
    # If not found, try to identify testable functions - especially render, listen, and register_*
    relevant_funcs = ["render", "listen", "register_api_endpoints", "register_descriptions"]
    
    for func_name in relevant_funcs:
        if hasattr(module, func_name):
            func = getattr(module, func_name)
            if callable(func):
                test_functions[func_name] = func
    
    return test_functions


def test_module(module_path: str, module_name: str) -> bool:
    """
    Test a specific module.
    
    Args:
        module_path (str): Path to the module file
        module_name (str): Name of the module
        
    Returns:
        bool: True if test was successful, False otherwise
    """
    print(f"\nTesting module: {module_name}")
    print("=" * 50)
    
    # Load the module
    module = load_module(module_path, module_name)
    if not module:
        return False
    
    # Find testable functions
    test_functions = find_testable_functions(module)
    
    if not test_functions:
        print(f"No testable functions found in {module_name}")
        return False
    
    # Try to call the test function if it exists
    if "test" in test_functions:
        try:
            print(f"Calling dedicated test function for {module_name}")
            test_functions["test"]()
            print(f"✅ Test for {module_name} completed successfully")
            return True
        except Exception as e:
            print(f"❌ Error testing {module_name}: {e}")
            return False
    
    # Otherwise, print available functions that could be used for testing
    print(f"No dedicated test function found for {module_name}.")
    print("The following functions could potentially be used for testing:")
    for func_name in test_functions:
        print(f"  - {func_name}")
    
    # For now, we won't try to automatically call these functions as they might have side effects
    print(f"⚠️ Manual testing required for {module_name}")
    return False


def main():
    """Main function to discover and test layout modules."""
    layout_files = discover_layout_modules()
    
    if not layout_files:
        print("No layout modules found!")
        return
    
    # Create a sorted list of modules
    module_names = sorted(layout_files.keys())
    
    print("AudioLab Layout Module Tester")
    print("=" * 50)
    print("Available modules:")
    print("1. All modules")
    
    for i, name in enumerate(module_names, 2):
        print(f"{i}. {name}")
    
    choice = input("\nEnter number of module to test (or 'q' to quit): ")
    
    if choice.lower() == 'q':
        return
    
    try:
        choice = int(choice)
        if choice == 1:
            # Test all modules
            print("\nTesting all modules...")
            for name, path in layout_files.items():
                test_module(path, name)
        elif 2 <= choice <= len(module_names) + 1:
            # Test specific module
            module_index = choice - 2
            module_name = module_names[module_index]
            module_path = layout_files[module_name]
            test_module(module_path, module_name)
        else:
            print("Invalid selection.")
    except ValueError:
        print("Please enter a valid number.")


if __name__ == "__main__":
    main() 