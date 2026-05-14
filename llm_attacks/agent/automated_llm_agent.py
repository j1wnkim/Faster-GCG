import logging
from typing import Literal

from .base import *
from .load_template import *
from .render_template import *

ChatTemplate = List[EasyInputMessageParam]


class AsyncAutomatedLLMAgent:
    """Automated LLM agent filled with template.

    Args:
        model_id (str):
            Huggingface model id or local path.
        template (dict[list[dict[str, Any]]] | None):
            Template should be a dict with the sole key of 'prompts'. template['prompts']
            is in the format of a chat. The first turn may be system prompt.
            All other turns should be user prompt. Default: None.
        template_path (str | None):
            Template path to load template. Default: None.
        seed (int | None):
            Random seed for inference. Default: None.
    """

    def __init__(
        self,
        engine: AsyncBaseAgent,
        template: Optional[ChatTemplate] = None,
        template_path: Optional[str] = None,
    ):
        self.engine = engine
        self.output_texts: list[str] = []
        """Output texts of all turns."""
        self.reasoning_texts: list[str] = []
        """Reasoning texts of all turns."""
        # template要求是list of message，[{role，content},{role，content},...]
        # template_path则多了一个prompts的key{prompts: [{role，content},{role，content},...]}
        if template is None:
            if template_path is not None:
                template: ChatTemplate = load_template(template_path)["prompts"]
        elif template_path is not None:
            logging.warning(
                "[AsyncAutomatedLLMAgent.init] template overrides template_path"
            )
        # conversation template
        self.conv_template = template or []
        if len(self.conv_template) > 0 and self.conv_template[0]["role"] == "system":
            self.engine.system_prompt = self.conv_template[0]["content"]

    @property
    def chat(self):
        return self.engine.chat

    @chat.setter
    def chat(self, value: ResponseInputParam):
        self.engine.chat = value

    @property
    def total_tokens(self):
        return self.engine.total_tokens

    @property
    def prompt_tokens(self):
        return self.engine.prompt_tokens

    @property
    def response_tokens(self):
        return self.engine.response_tokens

    def set_template(self, template: list[str | EasyInputMessageParam]):
        """Set conversation template.

        Args:
            template (list[str | EasyInputMessageParam]):
                Conversation template.

        Returns:
            self: Self.
        """
        self.conv_template = []
        for turn in template:
            if isinstance(turn, str):
                self.conv_template.append(
                    EasyInputMessageParam(content=turn, role="user")
                )
            else:
                self.conv_template.append(turn)
        if len(self.conv_template) > 0 and self.conv_template[0]["role"] == "system":
            self.engine.system_prompt = self.conv_template[0]["content"]
        return self

    async def run(
        self, values: Optional[dict[str, str]] = None, max_tokens=-1, **kwargs
    ):
        """Run full conversation.

        Args:
            values (dict[str, str] | None):
                Template values.
            max_tokens (int):
                Max new tokens. Default: -1.
            **kwargs:
                All args passed to generate response.
        """
        self.engine.new_chat()
        self.output_texts.clear()
        self.reasoning_texts.clear()

        if len(self.conv_template) == 0:
            logging.warning(
                "[AsyncAutomatedLLMAgent.run] No template or template_path specified, using empty template"
            )

        for turn in self.conv_template:
            if turn["role"] == "system":
                # X-Teaming needs template rendering for system prompt
                # Manually inject system prompt is okay.
                self.engine.chat.append(copy.deepcopy(turn))
                if values:
                    # Render template
                    content = self.engine.chat[-1]["content"]
                    assert self.engine.chat[-1]["role"] == "system"
                    if not isinstance(content, str):
                        raise NotImplementedError(
                            "Template rendering for non-string not implemented yet"
                        )
                    content = render_template(content, values)
                    assert content is not None, (
                        f"content is None after rendering:{self.engine.chat[-1]['content']}"
                    )
                    self.engine.chat[-1]["content"] = content
            elif turn["role"] == "user":
                self.engine.chat.append(copy.deepcopy(turn))
                if values:
                    # Render template
                    content = self.engine.chat[-1]["content"]
                    if not isinstance(content, str):
                        raise NotImplementedError(
                            "Template rendering for non-string not implemented yet"
                        )
                    content = render_template(content, values)
                    assert content is not None, (
                        f"content is None after rendering:{self.engine.chat[-1]['content']}"
                    )
                    self.engine.chat[-1]["content"] = content
                await self.engine.generate_response(max_tokens=max_tokens, **kwargs)
                self.output_texts.append(self.engine.output_text)
                self.reasoning_texts.append(self.engine.reasoning_text)
            else:
                raise RuntimeError(
                    f"role '{turn['role']}' should not exist in conversation template"
                )
