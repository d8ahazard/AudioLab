# AudioLab Coding Conventions

This document outlines coding conventions and best practices for the AudioLab project to ensure consistent and maintainable code.

## Python Code Style

### General Style
- Follow PEP 8 guidelines for Python code
- Use 4 spaces for indentation (no tabs)
- Maximum line length of 100 characters
- Use descriptive variable and function names
- Add docstrings to all classes and functions
- Use type hints for function parameters and return values

### Import Conventions
- Group imports in the following order:
  1. Standard library imports
  2. Third-party library imports
  3. Local application imports
- Use absolute imports for application modules

### Class Structure
- Use camelCase for method names
- Use PascalCase for class names
- Use snake_case for variables and function parameters
- Private methods and variables should be prefixed with underscore (_)

## UI Component Naming

### Element IDs and Classes
- Every UI element should have:
  - `elem_id` and `key` attributes with a prefix matching the tab name
  - `elem_classes="hintitem"` for tooltip functionality
  - Descriptions registered via `register_descriptions()`

### Naming Conventions
- Use descriptive names for UI elements that indicate their purpose
- Button names should use action verbs (e.g., "generate_btn", "export_btn")
- Input field names should describe the data they contain (e.g., "prompt_input", "model_dropdown")
- Use suffixes to indicate element type (e.g., "_btn", "_input", "_dropdown", "_slider")

## Wrapper Conventions

### BaseWrapper Implementation
- All wrappers must inherit from `BaseWrapper`
- Must implement required methods:
  - `process_audio()`: Core processing logic
  - `register_api_endpoint()`: API registration
  - `render_options()`: UI component rendering
  - `validate_args()`: Parameter validation

### Parameter Structure
- Define `allowed_kwargs` with complete parameter metadata:
  - `type`: Parameter data type
  - `default`: Default value
  - `min/max`: Range limits for numeric parameters
  - `choices`: List of allowed values for selection parameters
  - `render`: Whether to show in UI
  - `required`: Whether parameter is required

### Priority Conventions
- Follow the established priority ranges for wrappers:
  - Input Processors: 100-199
  - Transformation Processors: 200-299
  - Combination Processors: 300-399
  - Export Processors: 400-499

## API Conventions

### Endpoint Naming
- Use RESTful naming conventions for endpoints
- Follow the pattern `/api/v1/{feature}/{action}`
- Use plural nouns for resource collections

### Parameter Validation
- Use Pydantic models for request validation
- Document all parameters with types and constraints
- Return helpful error messages for invalid requests

### Response Format
- Use consistent JSON response format
- Include status code and message in all responses
- For file downloads, use appropriate content types and headers

## Error Handling

### Exception Handling
- Use specific exceptions rather than generic ones
- Handle exceptions at the appropriate level
- Log exceptions with detailed context
- Return user-friendly error messages

### Logging
- Use the logging module for all log messages
- Include appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- Log sufficient context to debug issues 