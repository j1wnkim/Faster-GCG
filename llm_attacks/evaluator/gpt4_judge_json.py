from .llm_judge_json import *


@register_evaluator("gpt4_judge_json")
class GPT4JsonEvaluator(LLMJudgeJsonEvaluator):
    """GPT-4 Judge Evaluator with json output format.

    ```plaintext
    @article{qi2023fine,
        title={Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!},
        author={Qi, Xiangyu and Zeng, Yi and Xie, Tinghao and Chen, Pin-Yu and Jia, Ruoxi and Mittal, Prateek and Henderson, Peter},
        journal={arXiv preprint arXiv:2310.03693},
        year={2023}
    }
    ```
    """

    def __init__(self, model_config: dict, max_retries=3):
        """Initialize LLM judge evaluator.

        Args:
            model_config (dict): Model configuration to use for evaluation
        """
        super().__init__(
            model_config=model_config,
            template_path=EVAL_DIR / "gpt4_judge.yaml",
            max_retries=max_retries,
        )
