from .base import Reward
from .exact_match import ExactMatchReward
from .numerical_tolerance import NumericalToleranceReward
from .code_execution import CodeExecutionReward
from .llm_judge import LLMJudgeReward

REWARD_REGISTRY = {
    "exact_match": ExactMatchReward,
    "numerical_tolerance": NumericalToleranceReward,
    "code_execution": CodeExecutionReward,
    "llm_judge": LLMJudgeReward,
}

__all__ = [
    "Reward",
    "ExactMatchReward",
    "NumericalToleranceReward",
    "CodeExecutionReward",
    "LLMJudgeReward",
    "REWARD_REGISTRY",
]
