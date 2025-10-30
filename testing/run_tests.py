#!/usr/bin/env python3
"""
AudioLab Testing Framework Runner

This script provides a comprehensive test runner for the AudioLab testing framework.
It supports running different types of tests with various configurations.
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Callable

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Note: Removed imports that cause circular dependency
# Will implement simple validation directly in this file


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('testing/reports/test_execution.log')
        ]
    )


def discover_embedded_tests() -> Dict[str, List[Callable]]:
    """
    Discover test functions embedded in AudioLab modules.

    Returns:
        Dictionary mapping module names to lists of test functions
    """
    test_functions = {}

    # Search in layouts directory
    layouts_dir = project_root / "layouts"
    if layouts_dir.exists():
        for py_file in layouts_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue

            module_name = py_file.stem
            tests = _find_tests_in_module(py_file, module_name)
            if tests:
                test_functions[f"layouts.{module_name}"] = tests

    # Search in modules directory
    modules_dir = project_root / "modules"
    if modules_dir.exists():
        for module_path in modules_dir.rglob("*.py"):
            # Skip __pycache__ and __init__.py files
            if "__pycache__" in str(module_path) or module_path.name.startswith("__"):
                continue

            # Convert path to module name
            rel_path = module_path.relative_to(modules_dir)
            module_name = str(rel_path).replace(os.sep, ".").replace(".py", "")

            tests = _find_tests_in_module(module_path, module_name)
            if tests:
                test_functions[f"modules.{module_name}"] = tests

    return test_functions


def _find_tests_in_module(module_path: Path, module_name: str) -> List[Callable]:
    """Find test functions in a specific module file."""
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return []

        module = importlib.util.module_from_spec(spec)

        # Add importlib to the module's globals so it can access importlib functions
        # This is needed because some modules might try to use importlib themselves
        module.__dict__.update({
            'importlib': importlib,
            'importlib_util': importlib.util,
            'inspect': inspect
        })

        spec.loader.exec_module(module)

        # Find test functions
        test_functions = []
        for name, obj in inspect.getmembers(module):
            if (inspect.isfunction(obj) and
                name.startswith('test') and
                not name.startswith('test_') and
                callable(obj)):
                test_functions.append(obj)

        return test_functions

    except Exception as e:
        logging.warning(f"Could not load module {module_name}: {e}")
        return []


def validate_environment() -> bool:
    """Validate that the testing environment is properly set up."""
    required_paths = [
        "layouts",
        "modules",
        "testing"
    ]

    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)

    if missing_paths:
        print(f"[ERROR] Test environment validation failed. Missing: {missing_paths}")
        return False

    print("[SUCCESS] Test environment validation passed")
    return True


def create_test_samples() -> None:
    """Create test audio samples if they don't exist."""
    samples_dir = Path("testing/config/audio_samples")
    if not samples_dir.exists():
        print("Creating test audio samples...")
        try:
            # Create directory
            samples_dir.mkdir(parents=True, exist_ok=True)

            # Create a simple test audio file
            import numpy as np
            try:
                import soundfile as sf

                # Generate sine wave
                sample_rate = 22050
                duration = 2.0
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                audio_data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

                # Save test file
                test_file = samples_dir / "test_sample.wav"
                sf.write(str(test_file), audio_data, sample_rate)
                print("[SUCCESS] Test audio samples created")
            except ImportError:
                print("[WARNING] soundfile not available, skipping test sample creation")
        except Exception as e:
            print(f"[WARNING] Could not create test samples: {e}")


def run_embedded_tests(test_functions: Dict[str, List[Callable]], args: argparse.Namespace) -> int:
    """Run embedded test functions."""
    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    print("\n" + "="*60)
    print("RUNNING EMBEDDED TESTS")
    print("="*60)

    for module_name, tests in test_functions.items():
        if not tests:
            continue

        print(f"\n[MODULE] {module_name}")
        print("-" * 50)

        for test_func in tests:
            total_tests += 1
            test_name = test_func.__name__

            try:
                print(f"  [TEST] Running {test_name}...")

                # Run the test function
                result = test_func()

                if result is True or result is None:
                    print(f"    [PASS] {test_name} PASSED")
                    passed_tests += 1
                else:
                    print(f"    [FAIL] {test_name} FAILED (returned: {result})")
                    failed_tests += 1

            except Exception as e:
                print(f"    [ERROR] {test_name} ERROR: {e}")
                failed_tests += 1

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")

    if args.verbose:
        success_rate = (passed_tests/total_tests)*100 if total_tests > 0 else 0
        print(f"\n[STATS] Success Rate: {success_rate:.1f}%")

    return 0 if failed_tests == 0 else 1


def run_layout_tests(args: argparse.Namespace) -> int:
    """Run tests embedded in layout modules."""
    print("\n" + "="*60)
    print("RUNNING LAYOUT TESTS")
    print("="*60)

    test_functions = discover_embedded_tests()
    layout_tests = {k: v for k, v in test_functions.items() if k.startswith('layouts.')}

    return run_embedded_tests(layout_tests, args)


def run_module_tests(args: argparse.Namespace) -> int:
    """Run tests embedded in core modules."""
    print("\n" + "="*60)
    print("RUNNING MODULE TESTS")
    print("="*60)

    test_functions = discover_embedded_tests()
    module_tests = {k: v for k, v in test_functions.items() if k.startswith('modules.')}

    return run_embedded_tests(module_tests, args)


def run_all_embedded_tests(args: argparse.Namespace) -> int:
    """Run all embedded tests."""
    test_functions = discover_embedded_tests()
    return run_embedded_tests(test_functions, args)


def run_pytest(args: List[str]) -> int:
    """Run pytest with specified arguments."""
    cmd = [sys.executable, "-m", "pytest"] + args

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 130
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def run_unit_tests(args: argparse.Namespace) -> int:
    """Run unit tests."""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)

    pytest_args = ["testing/unit/"]

    if args.coverage:
        pytest_args.extend([
            "--cov=audiolab",
            "--cov-report=html:testing/reports/coverage/html",
            "--cov-report=xml:testing/reports/coverage/coverage.xml",
            "--cov-report=json:testing/reports/coverage/coverage.json"
        ])

    if args.verbose:
        pytest_args.append("-v")

    if args.parallel:
        pytest_args.extend(["-n", str(args.parallel_workers)])

    if args.timeout:
        pytest_args.extend(["--timeout", str(args.timeout)])

    if args.specific_test:
        pytest_args.append(args.specific_test)

    return run_pytest(pytest_args)


def run_integration_tests(args: argparse.Namespace) -> int:
    """Run integration tests."""
    print("\n" + "="*60)
    print("RUNNING INTEGRATION TESTS")
    print("="*60)

    pytest_args = ["testing/integration/"]

    if args.coverage:
        pytest_args.extend([
            "--cov=audiolab",
            "--cov-report=html:testing/reports/coverage/integration_html",
            "--cov-append"
        ])

    if args.verbose:
        pytest_args.append("-v")

    if args.timeout:
        pytest_args.extend(["--timeout", str(args.timeout)])

    return run_pytest(pytest_args)


def run_system_tests(args: argparse.Namespace) -> int:
    """Run system/end-to-end tests."""
    print("\n" + "="*60)
    print("RUNNING SYSTEM TESTS")
    print("="*60)

    pytest_args = ["testing/system/"]

    if args.coverage:
        pytest_args.extend([
            "--cov=audiolab",
            "--cov-report=html:testing/reports/coverage/system_html",
            "--cov-append"
        ])

    if args.verbose:
        pytest_args.append("-v")

    if args.timeout:
        pytest_args.extend(["--timeout", str(args.timeout)])

    return run_pytest(pytest_args)


def run_all_tests(args: argparse.Namespace) -> int:
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60)

    exit_codes = []

    # Run unit tests
    exit_codes.append(run_unit_tests(args))

    # Run integration tests
    exit_codes.append(run_integration_tests(args))

    # Run system tests
    exit_codes.append(run_system_tests(args))

    # Return worst exit code
    return max(exit_codes) if exit_codes else 0


def generate_reports(args: argparse.Namespace) -> None:
    """Generate test reports."""
    print("\n" + "="*60)
    print("GENERATING TEST REPORTS")
    print("="*60)

    reports_dir = Path("testing/reports")
    reports_dir.mkdir(exist_ok=True)

    # Generate coverage report if coverage data exists
    coverage_html = reports_dir / "coverage" / "index.html"
    if coverage_html.exists():
        print(f"✓ Coverage report available at: {coverage_html}")
    else:
        print("⚠ No coverage report found")

    # Generate test summary
    summary_file = reports_dir / "test_summary.txt"
    try:
        with open(summary_file, 'w') as f:
            f.write("AudioLab Test Execution Summary\\n")
            f.write("="*40 + "\\n")
            f.write(f"Execution Time: {os.times()}\\n")
            f.write(f"Python Version: {sys.version}\\n")
            f.write(f"Platform: {sys.platform}\\n")

        print(f"✓ Test summary written to: {summary_file}")
    except Exception as e:
        print(f"⚠ Could not generate summary: {e}")


def cleanup(args: argparse.Namespace) -> None:
    """Clean up test artifacts."""
    print("\n" + "="*60)
    print("CLEANING UP TEST ARTIFACTS")
    print("="*60)

    cleanup_dirs = [
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        "testing/reports/coverage",
        "testing/config/audio_samples"
    ]

    for cleanup_pattern in cleanup_dirs:
        try:
            if cleanup_pattern.startswith("testing/"):
                path = Path(cleanup_pattern)
                if path.exists():
                    if path.is_file():
                        path.unlink()
                    else:
                        import shutil
                        shutil.rmtree(path)
                    print(f"[SUCCESS] Cleaned: {cleanup_pattern}")
        except Exception as e:
            print(f"[WARNING] Could not clean {cleanup_pattern}: {e}")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="AudioLab Testing Framework Runner")

    # Test type selection
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--layouts", action="store_true",
        help="Run tests embedded in layout modules only"
    )
    test_group.add_argument(
        "--modules", action="store_true",
        help="Run tests embedded in core modules only"
    )
    test_group.add_argument(
        "--all", action="store_true",
        help="Run all embedded tests"
    )

    # Configuration options
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )

    # Utility options
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate test environment"
    )
    parser.add_argument(
        "--create-samples", action="store_true",
        help="Create test audio samples"
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Clean up temporary test files"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Validate environment
    if not validate_environment():
        if args.validate_only:
            sys.exit(1)
        print("[WARNING] Continuing despite environment issues...")

    # Handle utility operations
    if args.validate_only:
        print("[SUCCESS] Validation complete")
        sys.exit(0)

    if args.create_samples:
        create_test_samples()
        sys.exit(0)

    if args.cleanup:
        cleanup(args)
        sys.exit(0)

    # Determine which tests to run
    if args.all or (not args.layouts and not args.modules):
        # Default behavior - run all embedded tests
        exit_code = run_all_embedded_tests(args)
    elif args.layouts:
        exit_code = run_layout_tests(args)
    elif args.modules:
        exit_code = run_module_tests(args)
    else:
        exit_code = run_all_embedded_tests(args)

    # Exit with appropriate code
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
