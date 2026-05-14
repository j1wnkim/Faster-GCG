"""OpenAI Completion agent.
Works with OpenAI Completion API.

Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
"""

import logging
import os
from typing import List, Optional

import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletionMessage, Choice

from llm_attacks.utils import parse_llm_reasoning

from .base import *

_logger = logging.getLogger(__name__)


class AsyncOpenAICompletionAgent(AsyncBaseAgent):
    """Async LLM agent using OpenAI Completion API.

    Args:
        model_id (str):
            Model ID exposed by the API.
        api_key (str):
            API key.
        server_url (str):
            API base url, e.g. https://api.openai.com/v1
        system_prompt (str | None):
            System prompt. None to disable.
        kwargs (dict | None):
            Sampling params or else for all generations.
    """

    chat: List[ChatCompletionMessageParam]

    def __init__(
        self,
        model_id: str = "gpt-5",
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(system_prompt=system_prompt)
        self.model_id = model_id

        self.client = openai.AsyncOpenAI(
            # Compatibility with vLLM. vLLM does not need a key, but AsyncOpenAI requires
            # a key.
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy-key"),
            base_url=base_url,
            max_retries=10,
        )

        self.request_params = kwargs
        """Default request params from initialization."""

        # Init a chat
        self.new_chat()
        self.chat_completion: Optional[ChatCompletion] = None

    @override
    async def generate_response(self, max_tokens=-1, **kwargs) -> str:
        """Generate a response via OpenAI Responses API."""
        # request params
        params = copy.copy(self.request_params)
        params.update(kwargs)
        if max_tokens > 0:
            params.update(max_completion_tokens=max_tokens)

        # Use OpenAI's async responses API
        completion = await self.client.chat.completions.create(
            model=self.model_id,
            messages=self.chat,
            stream=False,
            **params,
        )
        self.chat_id = completion.id

        if not len(completion.choices):
            _logger.warning("Model returns no choice")
            # Mock an empty choice
            completion.choices.append(
                Choice(
                    index=0,
                    finish_reason="stop",
                    message=ChatCompletionMessage(role="assistant"),
                )
            )

        choice = completion.choices[0]
        message = completion.choices[0].message
        if message.refusal:
            self.refused = True
            self.output_text = message.refusal
        elif choice.finish_reason == "refusal":
            # Claude
            self.refused = True
            self.output_text = "THE MODEL REFUSED TO RESPONSE."
        else:
            self.refused = False
            self.output_text = message.content or ""

        self.output_text, reasoning_text = parse_llm_reasoning(self.output_text)
        self.reasoning_text = reasoning_text or ""

        message_dict = message.to_dict()
        if "annotations" in message_dict:
            del message_dict["annotations"]
        self.chat.append(message_dict)

        # Store the latest response object
        self.chat_completion = completion
        # Usage stat
        self.total_tokens = completion.usage.total_tokens
        self.response_tokens += completion.usage.completion_tokens
        return self.output_text
