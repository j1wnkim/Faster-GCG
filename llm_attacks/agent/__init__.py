"""Build stateful LLM agent."""

from .automated_llm_agent import AsyncAutomatedLLMAgent, ChatTemplate
from .base import *
from .load_template import load_template
from .openai_completion_agent import AsyncOpenAICompletionAgent
from .openai_response_agent import AsyncOpenAIResponseAgent
from .render_template import render_template


def create_llm_agent(
    engine: str = "vllm",
    model="google/gemma-3-4b-it",
    server: str | None = None,
    seed: Optional[int] = None,
    async_: Optional[bool] = None,
    **kwargs,
) -> AsyncBaseAgent:
    if async_ == False:  # noqa: E712
        raise RuntimeError("Sync LLM interface has been removed")

    if engine in ("openai", "api"):
        return AsyncOpenAIResponseAgent(model_id=model, base_url=server, **kwargs)
    elif engine in ("openai_completion", "vllm", "vllm_server"):
        # Note: Latest vLLM supports response interface. For compatibility, we still use
        # completion interface.
        return AsyncOpenAICompletionAgent(model_id=model, base_url=server, **kwargs)
    else:
        raise ValueError(f"LLM serving engine '{engine}' not found")


def create_automated_llm_agent(
    engine: str = "vllm",
    model="google/gemma-3-4b-it",
    server: str | None = None,
    template: Optional[ChatTemplate] = None,
    template_path: Optional[str] = None,
    seed: Optional[int] = None,
    async_: Optional[bool] = None,
    **kwargs,
):
    engine_ = create_llm_agent(
        engine=engine,
        model=model,
        seed=seed,
        server=server,
        async_=async_,
        **kwargs,
    )
    return AsyncAutomatedLLMAgent(
        engine_, template=template, template_path=template_path
    )
