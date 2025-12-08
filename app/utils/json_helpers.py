"""
JSON parsing and manipulation utilities.
"""

import json
import re
from typing import Any, Dict, Optional


def strip_markdown_fences(text: str) -> str:
    """
    Remove markdown code fences from text.
    
    Handles:
    - ```json ... ```
    - ``` ... ```
    
    Args:
        text: Text potentially containing markdown fences
        
    Returns:
        str: Text with fences removed
    """
    text = text.strip()
    
    # Remove opening fence (```json or ```)
    text = re.sub(r'^```(?:json)?\n?', '', text)
    
    # Remove closing fence
    text = re.sub(r'\n?```$', '', text)
    
    return text.strip()


def safe_json_parse(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON with fallback.
    
    Args:
        text: JSON string
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        # Strip markdown fences first
        text = strip_markdown_fences(text)
        return json.loads(text)
    except json.JSONDecodeError:
        return default
    except Exception:
        return default


def parse_json_or_raise(text: str) -> Dict[str, Any]:
    """
    Parse JSON and raise descriptive error on failure.
    
    Args:
        text: JSON string
        
    Returns:
        Dict: Parsed JSON
        
    Raises:
        ValueError: If JSON is invalid with detailed error
    """
    try:
        # Strip markdown fences
        text = strip_markdown_fences(text)
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}\n"
            f"Context: {text[max(0, e.pos-50):e.pos+50]}"
        )


def pretty_print_json(data: Any, indent: int = 2) -> str:
    """
    Convert data to pretty-printed JSON string.
    
    Args:
        data: Data to serialize
        indent: Indentation level
        
    Returns:
        str: Pretty-printed JSON
    """
    return json.dumps(data, indent=indent, ensure_ascii=False)


def minify_json(data: Any) -> str:
    """
    Convert data to minified JSON string.
    
    Args:
        data: Data to serialize
        
    Returns:
        str: Minified JSON (no whitespace)
    """
    return json.dumps(data, separators=(',', ':'), ensure_ascii=False)


def deep_merge(base: Dict, update: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        update: Dictionary with updates
        
    Returns:
        Dict: Merged dictionary
    """
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text that may contain other content.
    
    Useful for extracting JSON from LLM responses that include
    explanations before/after the JSON.
    
    Args:
        text: Text containing JSON
        
    Returns:
        Dict or None: Extracted JSON or None if not found
    """
    # Try to find JSON object pattern
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    # Try to parse each potential JSON match
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


def validate_json_schema(data: Dict[str, Any], required_keys: list) -> tuple[bool, list]:
    """
    Validate that JSON contains required keys.
    
    Args:
        data: JSON data
        required_keys: List of required key names
        
    Returns:
        tuple: (is_valid, list_of_missing_keys)
    """
    missing_keys = [key for key in required_keys if key not in data]
    return len(missing_keys) == 0, missing_keys


def get_nested_value(data: Dict, key_path: str, default: Any = None) -> Any:
    """
    Get value from nested dictionary using dot notation.
    
    Example:
        data = {"user": {"profile": {"name": "John"}}}
        get_nested_value(data, "user.profile.name") => "John"
    
    Args:
        data: Dictionary to search
        key_path: Dot-separated path (e.g., "user.profile.name")
        default: Default value if path not found
        
    Returns:
        Value at path or default
    """
    keys = key_path.split('.')
    value = data
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError, IndexError):
        return default


def set_nested_value(data: Dict, key_path: str, value: Any) -> Dict:
    """
    Set value in nested dictionary using dot notation.
    
    Example:
        data = {}
        set_nested_value(data, "user.profile.name", "John")
        => {"user": {"profile": {"name": "John"}}}
    
    Args:
        data: Dictionary to modify
        key_path: Dot-separated path
        value: Value to set
        
    Returns:
        Dict: Modified dictionary
    """
    keys = key_path.split('.')
    current = data
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
    return data


def flatten_dict(data: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    Flatten nested dictionary.
    
    Example:
        flatten_dict({"a": {"b": {"c": 1}}})
        => {"a.b.c": 1}
    
    Args:
        data: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for keys
        
    Returns:
        Dict: Flattened dictionary
    """
    items = []
    
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def sanitize_json_string(text: str) -> str:
    """
    Clean up JSON string for safe parsing.
    
    - Removes control characters
    - Fixes common JSON issues
    - Strips markdown fences
    
    Args:
        text: Raw JSON string
        
    Returns:
        str: Sanitized JSON string
    """
    # Strip markdown fences
    text = strip_markdown_fences(text)
    
    # Remove control characters except newline and tab
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Fix common issues
    text = text.replace('\n', ' ')  # Remove newlines inside strings
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    
    return text.strip()


def is_valid_json(text: str) -> bool:
    """
    Check if string is valid JSON.
    
    Args:
        text: String to validate
        
    Returns:
        bool: True if valid JSON
    """
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def json_diff(old: Dict, new: Dict) -> Dict[str, Any]:
    """
    Find differences between two JSON objects.
    
    Args:
        old: Old version
        new: New version
        
    Returns:
        Dict: Differences with keys: added, removed, changed
    """
    diff = {
        "added": {},
        "removed": {},
        "changed": {}
    }
    
    # Find added and changed keys
    for key, value in new.items():
        if key not in old:
            diff["added"][key] = value
        elif old[key] != value:
            diff["changed"][key] = {"old": old[key], "new": value}
    
    # Find removed keys
    for key in old:
        if key not in new:
            diff["removed"][key] = old[key]
    
    return diff