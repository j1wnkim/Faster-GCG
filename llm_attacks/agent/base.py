import copy
from typing import List, Optional, overload, override

from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputMessageContentListParam,
    ResponseInputParam,
)


def unwrap_text_message(content: ResponseInputMessageContentListParam) -> str:
    """Unwrap text into pure text. Multi-modal contents are ignored."""
    texts: List[str] = []
    for segment in content:
        if "text" in segment["type"]:
            texts.append(segment["text"])
    return "".join(texts)


class AsyncBaseAgent:
    """Async base LLM agent.

    **About chat history:**
    We track chat history with the primary type of `openai.ResponseInputParam`, which is
    the most up-to-date type that supports rich contents. Note that we use the `*Param`,
    which are typed dicts, instead of the pydantic models that are used for output,
    because the SDK functions expect typed dicts as input params. You are always free to
    inherit the class and use another type for input.

    **Note: You should always clean up `chat_id` if you manually modify chat.**

    Args:
        system_prompt (str | None):
            System prompt. None to disable.
    """

    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt
        """System prompt. None if no system prompt is set."""
        self.chat: ResponseInputParam = []
        """Chat history."""
        self.chat_id: str | None = None
        """Chat ID from the API response. None if not set."""
        self.total_tokens = 0
        """Total tokens in current chat. Updated on `generate_response`."""
        self.response_tokens = 0
        """Response tokens in current chat. Updated on `generate_response`."""
        self.output_text = ""
        """Save the output text of the last turn."""
        self.reasoning_text = ""
        """Save the reasoning text of the last turn."""
        self.refused = False
        """Mark if refused."""

    @property
    def prompt_tokens(self):
        """Prompt tokens in current chat. Updated on `generate_response`."""
        return self.total_tokens - self.response_tokens

    def new_chat(self):
        """Start a new chat.

        Returns:
            self
        """
        self.chat = []
        self.chat_id = None
        self.total_tokens = 0
        self.response_tokens = 0
        self.output_text = ""
        self.reasoning_text = ""
        self.refused = False
        return self

    async def generate_response(self, max_tokens=-1, **kwargs) -> str:
        """Generate response with current chat.

        Args:
            max_tokens (int):
                Max new tokens. Default: -1.
            **kwargs:
                All kwargs passed to `SamplingParams`.

        Returns:
            str: answer text.
        """
        raise NotImplementedError

    async def ask(
        self,
        prompt: str | ResponseInputMessageContentListParam,
        max_tokens=-1,
        **kwargs,
    ) -> str:
        """Continues the chat and ask something to the model.

        Args:
            prompt (str | ResponseInputMessageContentListParam):
                User prompt. Accepts one of: (1) string; (2) a list of dict, each being a
                part of the message.
            max_tokens (int):
                Max new tokens. Default: -1.
            kwargs:
                All kwargs passed to generation.

        Returns:
            str: answer text.

        Raises:
            RuntimeError: when chat is in bad status.
        """
        if len(self.chat) == 0 and self.system_prompt is not None:
            # Lazy init system prompt
            self.chat.append(
                EasyInputMessageParam(role="system", content=self.system_prompt)
            )

        if len(self.chat) > 0 and self.chat[-1]["role"] == "user":
            # Ensure that the last turn has a response
            raise RuntimeError("Chat should be user-assistant in turn")

        # Append new turn and generate
        self.chat.append(EasyInputMessageParam(role="user", content=prompt))
        return await self.generate_response(max_tokens=max_tokens, **kwargs)
