import copy
import json
import re
from typing import Any, override

from llm_attacks.agent import create_llm_agent
from llm_attacks.utils.llm import decode_json_object_from_llm_output

from .base import *
from .registry import register_evaluator

with open(EVAL_DIR / "judge_templates.json", "r") as f:
    JUDGE_TEMPLATES = json.load(f)


@register_evaluator("llm_judge")
class LLMJudgeEvaluator(BaseEvaluator):
    """LLM-based judge evaluator for jailbreak detection.

    Uses an LLM to evaluate whether a model response to a forbidden prompt
    is harmful or follows safety guidelines. The judge provides both a
    numerical score and a reasoning explanation.
    """

    def __init__(self, model_config: dict, template: str, max_retries=3):
        """Initialize LLM judge evaluator.

        Args:
            model_config (dict): Model configuration to use for evaluation
            template (str): Template string for evaluation prompt
        """
        self.llm_agent = create_llm_agent(**model_config)
        self.template = template
        self.max_retries = max_retries

    def _format_prompt(
        self, forbidden_prompt: str | list[str], response: str | list[str]
    ) -> str:
        """Format the evaluation prompt.

        Args:
            template (str): The template string to use for the prompt
            forbidden_prompt (str): The user's potentially harmful prompt
            response (str): The model's response to evaluate

        Returns:
            str: Formatted prompt string
        """
        # Join multiple turns
        if not isinstance(forbidden_prompt, str):
            forbidden_prompt = "\n".join(forbidden_prompt)
        if not isinstance(response, str):
            response = "\n".join(response)
        return self.template.format(
            forbidden_prompt=forbidden_prompt, response=response
        )

    def _parse_judge_output(self, text: str) -> EvaluationResult | None:
        """Parse the judge's LLM output.

        Args:
            text (str): Raw LLM response text

        Returns:
            EvaluationResult | None: Parsed evaluation result or None if parsing fails
        """
        parsed = decode_json_object_from_llm_output(text, required_keys={"score"})
        if parsed is None:
            return None
        # Normalize to [0,1]
        score = (int(parsed["score"]) - 1) / 4
        return EvaluationResult(
            score=score,
            reason=parsed.get("reason"),
            raw=text,
        )

    @override
    async def evaluate(
        self, forbidden_prompt: str | list[str], response: str | list[str]
    ) -> EvaluationResult:
        """Evaluate a single prompt-response pair.

        Args:
            forbidden_prompt (str | list[str]): The user's potentially harmful prompt
            response (str | list[str]): The model's response to evaluate
            max_retries (int): Maximum number of retries on parse failure. Default: 3.

        Returns:
            EvaluationResult containing score and metadata

        Raises:
            RuntimeError: If the judge fails to parse after max_retries attempts
        """
        max_retries = self.max_retries
        for _ in range(max_retries):
            self.llm_agent.new_chat()
            llm_response = await self.llm_agent.ask(
                self._format_prompt(forbidden_prompt, response)
            )
            result = self._parse_judge_output(llm_response)
            if result is not None:
                return result
        raise RuntimeError(
            f"Failed to parse judge output after {max_retries} attempts. output: {llm_response}"
        )
