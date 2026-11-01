"""Numerical Tolerance (NT) reward.

Math: |extracted - gold| < atol after SymPy normalization, so "1/2" matches
      "0.5" and "sqrt(4)" matches "2"; strict string comparison if SymPy cannot
      parse either side.
Code: no separate tolerance-aware runner exists; every code sample is scored
      by ExactMatchReward (the unit-test pass rate).
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from .base import Reward
from . import _extract

# Bounds the subprocess-isolated SymPy evaluation in _sympy_verdict() below.
# Legitimate GSM8K/MATH-style symbolic comparisons finish in well under
# 100ms; this is generous headroom, not a tight budget.
_SYMPY_TIMEOUT_S = 3.0


# --------------------------------------------------------------------------- #
# Subprocess-isolated SymPy evaluation                                        #
# --------------------------------------------------------------------------- #
# SymPy evaluation can hang or exhaust memory on pathological but plausible model outputs
# (for example a power tower such as 2**2**2**2**2**2**2**2**2**2), and a hang would block the
# whole synchronous reward computation. It therefore runs in a subprocess with a hard
# wall-clock timeout. On timeout or any subprocess failure the result is treated as "SymPy
# could not determine this" and falls through to the bounded in-process fallbacks (plain
# float parsing, string normalisation), so pathological inputs fail closed after
# _SYMPY_TIMEOUT_S instead of hanging.
_SYMPY_VERDICT_HARNESS = r"""
import sys

extracted = __EXTRACTED_REPR__
gold = __GOLD_REPR__
atol = __ATOL_REPR__


def _latex_to_sympy_str(s):
    import re
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'((\1)/(\2))', s)
    s = re.sub(r'\\sqrt\s*\{([^{}]*)\}', r'sqrt(\1)', s)
    s = s.replace(r'\sqrt', 'sqrt')
    s = s.replace(r'\pi', 'pi').replace(r'\Pi', 'pi')
    s = s.replace(r'\cdot', '*').replace(r'\times', '*').replace(r'\div', '/')
    s = s.replace(r'\left', '').replace(r'\right', '')
    s = s.replace(r'\,', '').replace(r'\;', '').replace(r'\:', '').replace(r'\!', '')
    s = re.sub(r'\\(?:d|t)frac', r'\\frac', s)
    s = re.sub(r'\\mathrm\s*\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\text\s*\{([^{}]*)\}', r'\1', s)
    s = s.replace('^', '**')
    return s


def _try_parse(s):
    import sympy
    try:
        return sympy.sympify(s, evaluate=True)
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError):
        pass
    try:
        return sympy.sympify(_latex_to_sympy_str(s), evaluate=True)
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError):
        pass
    try:
        from sympy.parsing.latex import parse_latex
        return parse_latex(s)
    except Exception:
        return None


import sympy

a = _try_parse(extracted)
b = _try_parse(gold)
if a is None or b is None:
    print("UNPARSEABLE")
    sys.exit(0)

matched = False
try:
    if sympy.simplify(a - b) == 0:
        matched = True
except Exception:
    pass
if not matched:
    try:
        if abs(float(a) - float(b)) < atol:
            matched = True
    except (TypeError, ValueError):
        pass

print("MATCH" if matched else "NOMATCH")
"""


def _sympy_verdict(extracted: str, gold: str, atol: float) -> str:
    """Run the SymPy parse+compare step in a subprocess with a hard
    timeout. Returns "MATCH", "NOMATCH", or "UNPARSEABLE" (the last also
    covers timeout/crash -- same meaning as "SymPy couldn't determine
    this")."""
    code = (
        _SYMPY_VERDICT_HARNESS
        .replace("__EXTRACTED_REPR__", repr(extracted))
        .replace("__GOLD_REPR__", repr(gold))
        .replace("__ATOL_REPR__", repr(atol))
    )
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8",
        ) as fh:
            fh.write(code)
            tmp_path = fh.name
        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=_SYMPY_TIMEOUT_S,
                check=False,
            )
            out = (proc.stdout or "").strip()
            if out in ("MATCH", "NOMATCH", "UNPARSEABLE"):
                return out
            return "UNPARSEABLE"
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
    except subprocess.TimeoutExpired:
        return "UNPARSEABLE"
    except (OSError, ValueError):
        return "UNPARSEABLE"


def _math_passes(extracted: str | None, gold: str | None, atol: float) -> bool:
    """True iff extracted and gold are numerically within atol of each other.

    Layered fallbacks (most permissive last):
        1. SymPy parse (with LaTeX preprocessing) + simplify(a - b) == 0,
           OR SymPy parse + |a - b| < atol (numeric within tolerance) --
           both run subprocess-isolated with a hard timeout, see
           _sympy_verdict() above for why.
        2. Plain float(extracted) and float(gold) compared within atol
        3. Strict-EM via _extract.normalize on each
    """
    if extracted is None or gold is None:
        return False

    extracted = extracted.strip()
    gold = str(gold).strip()

    verdict = _sympy_verdict(extracted, gold, atol)
    if verdict == "MATCH":
        return True

    # Plain float compare (handles cases where sympify failed but they're floats)
    fa = _extract.parse_float_or_none(extracted)
    fb = _extract.parse_float_or_none(gold)
    if fa is not None and fb is not None:
        return abs(fa - fb) < atol

    # Last-ditch: normalized string equality (matches EM behaviour)
    return _extract.normalize(extracted) == _extract.normalize(gold)


class NumericalToleranceReward(Reward):
    def __init__(self, domain: str, atol: float = 1e-4):
        assert domain in ("math", "code"), domain
        self.domain = domain
        self.atol = float(atol)

    def compute(self, prompt: str, response: str, gold: dict) -> float:
        if self.domain == "math":
            gold_answer = (gold or {}).get("answer")
            if gold_answer is None:
                return 0.0
            extracted = _extract.extract_math_answer(response or "")
            return 1.0 if _math_passes(extracted, str(gold_answer), self.atol) else 0.0

        # Code: delegate to the exact-match (unit-test) behaviour.
        from .exact_match import ExactMatchReward
        return ExactMatchReward(domain="code").compute(prompt, response, gold)
