"""LLM-related utilities."""

import json
import re
from typing import Optional

# Note: json5 supports relaxed syntax (`pip install json5``)


def decode_json_object_from_llm_output(text: str, required_keys: Optional[set] = None):
    """Try to decode json object from LLM output.

    Args:
        text (str): LLM output text.
        required_keys (set | None): required keys.

    Returns:
        object (dict | None): dict for successfully decoded object dict, None if failed to
        decode.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        obj = json.loads(match.group(0))
        if not isinstance(obj, dict):
            return None
        if required_keys and not required_keys <= obj.keys():
            return None
    except json.JSONDecodeError:
        return None
    return obj


"""Here is an example of adding type to it.

from typing import Optional, Type, TypedDict, TypeVar, overload


T = TypeVar("T")


@overload
def decode_json_object_from_llm_output(
    text: str,
    required_type: Type[T],
) -> T | None: ...


@overload
def decode_json_object_from_llm_output(
    text: str,
) -> dict | None: ...

Use `required_keys = set(required_type.__annotations__.keys())` to extract keys.

But this is too complicated in use, so we don't add typing to it.
"""


def parse_qwen_reasoning(content: str):
    """Parse Qwen style reasoning.

    Args:
        content (str):
            Model response content.

    Returns:
        content,reasoning (tuple[str, str | None]):
            Content and reasoning content. Reasoning content may be none.
    """
    if "</think>" in content:
        # Parse Qwen reasoning
        # Extracted from Qwen3 chat template
        reasoning_content = (
            content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
        )
        content = content.split("</think>")[-1].lstrip("\n")
        return content, reasoning_content
    return content, None


def parse_llm_reasoning(content: str):
    """Parse LLM reasoning.

    Args:
        content (str):
            Model response content.

    Returns:
        content,reasoning (tuple[str, str | None]):
            Content and reasoning content. Reasoning content may be none.
    """
    return parse_qwen_reasoning(content)
