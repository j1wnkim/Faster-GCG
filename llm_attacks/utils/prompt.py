"""Minimal LLM prompt typing of openai chat completion api.

Note: This only provides type hint but not type validation.
Use `pydantic` if type validation is needed.

See: https://platform.openai.com/docs/guides/text?api-mode=chat
"""

import copy
from typing import Literal, Sequence, TypedDict


class MessagePartText(TypedDict):
    type: Literal["text"]
    text: str


class MessagePartImage(TypedDict):
    type: Literal["image"]
    image: str


MessagePart = MessagePartText | MessagePartImage
MessageContent = str | list[MessagePart]


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: MessageContent
    reasoning_content: str | None


Conversation = list[Message]
PromptLike = MessageContent | Conversation
BatchedPromptLike = list[PromptLike]


def unwrap_text_message(prompt: MessageContent):
    """Unwrap a text prompt to string.

    Args:
        prompt (MessageContent): Message.

    Returns:
        str: Text message.

    Raises:
        RuntimeError: when the input prompt is not text-only.
    """
    if isinstance(prompt, str):
        return prompt
    prompt_list: list[str] = []
    for p in prompt:
        if p["type"] != "text":
            raise RuntimeError(f"`unwrap_text_prompt` encounters {p['type']} prompt")
        prompt_list.append(p["text"])
    return "".join(prompt_list)


def is_conversation(prompt: PromptLike):
    """Decide if the prompt is a conversation.

    Args:
        prompt (PromptLike): A prompt of conversation.

    Returns:
        is_conversation (bool): True if the prompt is a conversation.
    """
    if isinstance(prompt, str):
        return False
    elif len(prompt) == 0:
        # There is no way to decide, we can just assume it is not a conversation.
        return False
    else:
        return "role" in prompt[0]


def make_conversation(
    prompt: PromptLike,
    system_prompt: MessageContent | None = None,
    copy_=False,
    pure_text=False,
    pure_multimodal=False,
) -> Conversation:
    """Make the prompt a conversation.

    Args:
        prompt (PromptLike): A prompt of conversation.
        system_prompt (Optional[MessageContent]): The system prompt to add.
        copy_ (bool): Set to True to ensure copy. Default: False.
        pure_text (bool): If set, each content is a string.
        pure_multimodal (bool): If set, each content is a dict.

    Returns:
        conversation (Conversation): The conversation.

    Raises:
        RuntimeError: If `pure_text=True`, and got a multi-modal `prompt`.
        ValueError: If `pure_text` and `pure_multimodal` are both set.
    """
    if pure_text and pure_multimodal:
        raise ValueError("`pure_text` and `pure_multimodal` cannot be both set")

    # Make conversation
    if is_conversation(prompt):
        conversation: Conversation = copy.deepcopy(prompt)
    elif system_prompt is not None:
        conversation: Conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        conversation: Conversation = [{"role": "user", "content": prompt}]

    # Cleanup pure_text or pure_multimodal
    if pure_text:
        for item in conversation:
            if not isinstance(item["content"], str):
                item["content"] = unwrap_text_message(item["content"])
    elif pure_multimodal:
        for item in conversation:
            if isinstance(item["content"], str):
                item["content"] = [{"type": "text", "text": item["content"]}]

    return conversation


def is_prompt_batched(prompt: PromptLike | BatchedPromptLike):
    """Check if the prompt is a batch."""
    if isinstance(prompt, str):
        return False
    if not isinstance(prompt, Sequence):
        return False
    if len(prompt) == 0:
        # 0-length batch
        return True
    return not isinstance(prompt[0], dict)
