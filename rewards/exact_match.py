"""Exact Match (EM) reward.

Math: extract the answer from \\boxed{} (with fallbacks), normalize, strict
      string match (no SymPy, no tolerance; those are NT and CE).
Code: unit-test pass rate on the generated code, delegated to
      CodeExecutionReward (character-exact match on code would be almost always 0).
"""
from __future__ import annotations

from .base import Reward
from . import _extract


# Aliases for the shared extraction helpers.
_extract_boxed = _extract.extract_boxed
_extract_math_answer = _extract.extract_math_answer
_normalize = _extract.normalize


class ExactMatchReward(Reward):
    def __init__(self, domain: str):
        assert domain in ("math", "code"), domain
        self.domain = domain

    def compute(self, prompt: str, response: str, gold: dict) -> float:
        if self.domain == "code":
            # EM for code is the unit-test pass rate.
            # The "exact match on code is trivially 0%" caveat means we
            # actually measure functional equivalence (test pass rate).
            from .code_execution import CodeExecutionReward
            return CodeExecutionReward(domain="code").compute(prompt, response, gold)

        # Math path
        gold_answer = (gold or {}).get("answer")
        if gold_answer is None:
            return 0.0
        extracted = _extract.extract_math_answer(response or "")
        if extracted is None:
            return 0.0
        return 1.0 if _extract.normalize(extracted) == _extract.normalize(str(gold_answer)) else 0.0
