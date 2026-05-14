from typing import override

from .base import *
from .registry import *


@register_evaluator("ensemble")
class EvaluatorEnsemble(BaseEvaluator):
    def __init__(self, evaluators: list[BaseEvaluator | dict]):
        self.evaluators: list[BaseEvaluator] = []
        for evaluator in evaluators:
            if isinstance(evaluator, dict):
                evaluator = get_evaluator(**evaluator)
            self.evaluators.append(evaluator)

    @override
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

        Raises:
            RuntimeError: If any of the evaluators raises an error
        """
        results: list[EvaluationResult] = []
        reason = ""
        raw = ""
        for i, evaluator in enumerate(self.evaluators):
            # May raise an error
            result = await evaluator.evaluate(forbidden_prompt, response, **kwargs)
            results.append(result)
            if result.reason:
                reason += f"Evaluator {i + 1} (score {result.score:.2f}):\n{result.reason}\n\n"
            if result.raw:
                raw += (
                    f"Evaluator {i + 1} (score {result.score:.2f}):\n{result.raw}\n\n"
                )
        # Use average score
        return EvaluationResult(
            score=sum(result.score for result in results) / len(results),
            reason=reason.strip() or None,
            raw=raw.strip() or None,
        )
