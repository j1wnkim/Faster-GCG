import os
import re
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

# Regex pattern to match ${VAR} format
_ENV_VAR_PATTERN = re.compile(r"\$\{(.+?)\}")


def substitute_env_vars(obj: T) -> T:
    """
    Recursively replace environment variables in a YAML-loaded object.

    Supported environment variable formats:
    - ${VAR_NAME}: Standard format for environment variables

    Behavior:
    - If an environment variable is found, it is substituted
    - If an environment variable is not found, the original string (e.g., ${VAR_NAME}) is preserved

    Args:
        obj (Any):
            YAML-loaded object (dict, list, string, etc.)

    Returns:
        obj: Modified object with environment variables substituted
    """

    def replace_env_var(match: re.Match):
        """Replace a single environment variable match."""
        # Strip whitespace around variable name
        var_name = match.group(1).strip()
        # Fallback to original if not found
        return os.environ.get(var_name, match.group(0))

    if isinstance(obj, dict):
        # Recursively process each value in the dictionary
        return {key: substitute_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Recursively process each item in the list
        return [substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Replace environment variables in strings
        return re.sub(_ENV_VAR_PATTERN, replace_env_var, obj)
    else:
        # For non-string, non-collection types, return as-is
        return obj


def safe_load_with_env(stream: Any) -> Any:
    """YAML safe load with environment variable substitution."""
    # Load the original object
    obj = yaml.safe_load(stream)
    # Recursively substitute environment variables
    return substitute_env_vars(obj)
