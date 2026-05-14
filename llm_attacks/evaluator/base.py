from pathlib import Path
from typing import override

import pydantic

EVAL_DIR = Path(__file__).parent.resolve() / "eval_files"


class EvaluationResult(pydantic.BaseModel):
    """Result from an evaluator."""

    score: float
    """Score in range [0, 1], where 0 is safe/refusal and 1 is harmful/jailbroken."""
    reason: str | None = None
    """Human-readable explanation of the evaluation."""
    raw: str | None = None
    """Raw output from the evaluator (e.g., LLM response)."""


class BaseEvaluator:
    """Base class for all evaluators.

    Evaluators assess whether a model response to a prompt is harmful or follows safety guidelines.
    """

    async def evaluate(
        self,
        forbidden_prompt: str | list[str],
        response: str | list[str],
        **kwargs,
    ) -> EvaluationResult:
        """Evaluate a single prompt-response pair.

        Args:
            forbidden_prompt (str | list[str]):
                The user's potentially harmful prompt. A list corresponds to multiple turns.
            response (str | list[str]):
                The model's response to evaluate. A list corresponds to multiple turns.
            **kwargs: Additional evaluator-specific parameters

        Returns:
            EvaluationResult containing score and metadata
        """
        raise NotImplementedError

    async def __call__(
        self,
        forbidden_prompt: str | list[str],
        response: str | list[str],
        **kwargs,
    ) -> EvaluationResult:
        """Evaluate a single prompt-response pair.

        Args:
            forbidden_prompt (str | list[str]):
                The user's potentially harmful prompt. A list corresponds to multiple turns.
            response (str | list[str]):
                The model's response to evaluate. A list corresponds to multiple turns.
            **kwargs: Additional evaluator-specific parameters

        Returns:
            EvaluationResult containing score and metadata
        """
        return await self.evaluate(forbidden_prompt, response, **kwargs)
