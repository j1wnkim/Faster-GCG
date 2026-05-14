"""OpenAI Response agent.
Works with OpenAI Responses API.

Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
"""

import os

import openai
from openai.types.responses import Response

from llm_attacks.utils import parse_llm_reasoning

from .base import *


class AsyncOpenAIResponseAgent(AsyncBaseAgent):
    """Async LLM agent using OpenAI Responses API.

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
        self.response: Optional[Response] = None

    @override
    async def generate_response(self, max_tokens=-1, **kwargs) -> str:
        """Generate a response via OpenAI Responses API."""
        # request params
        params = copy.copy(self.request_params)
        params.update(kwargs)
        if max_tokens > 0:
            params.update(max_output_tokens=max_tokens)

        if self.chat_id:
            # Multi-turn chat. The last item is new user prompt.
            input = self.chat[-1:]
        else:
            # The first turn.
            input = self.chat

        # Use OpenAI's async responses API
        response = await self.client.responses.create(
            model=self.model_id,
            input=input,
            previous_response_id=self.chat_id,
            stream=False,
            **params,
        )

        # Gather output and convert to params
        output_texts: List[str] = []
        reasoning_texts: List[str] = []
        refused = False
        for output in response.output:
            # Use to_dict to dump to dict
            self.chat.append(output.to_dict())

            # Extract text and reasoning
            if output.type == "message":
                for content in output.content:
                    if content.type == "output_text":
                        output_texts.append(content.text)
                    elif content.type == "refusal":
                        output_texts.append(content.refusal)
                        refused = True
            elif output.type == "reasoning" and output.content:
                for content in output.content:
                    if content.type == "reasoning_text":
                        reasoning_texts.append(content.text)

        self.output_text = "".join(output_texts)
        self.reasoning_text = "".join(reasoning_texts)
        self.refused = refused

        # Compatibility with vLLM.
        if not self.reasoning_text:
            self.output_text, reasoning_text = parse_llm_reasoning(self.output_text)
            self.reasoning_text = reasoning_text or ""

        # Store the latest response object
        self.response = response
        self.chat_id = response.id
        # Usage stat
        self.total_tokens = response.usage.total_tokens
        self.response_tokens += response.usage.output_tokens
        return self.output_text
