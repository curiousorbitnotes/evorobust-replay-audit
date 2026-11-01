"""Shared extraction + normalization helpers used by every Reward subclass.

Math answer extraction order (any reward that needs the model's "final answer"):
    1. The contents of the LAST balanced \\boxed{...} in the response.
    2. GSM8K-style "#### <answer>" line.
    3. "the (final) answer is <answer>" sentence.
    4. The last numeric token anywhere in the response.

Code extraction:
    1. Triple-backtick code fence (```python ... ``` or ``` ... ```).
    2. Plain text (caller's risk if not actually Python).

Normalization:
    Strip whitespace, $ and trailing periods, drop commas and spaces; collapse "72.0" -> "72".
"""
from __future__ import annotations

import re

_BOXED_OPEN = "\\boxed{"

_HASHHASH = re.compile(r"####\s*([^\n]+?)\s*$", re.MULTILINE)
_ANSWER_IS = re.compile(r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+([^\n]+)")
_LAST_NUMBER = re.compile(r"-?\d+(?:[\.,]\d+)*")
_CODE_FENCE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...} in text if its braces balance, else None."""
    if not text:
        return None
    idx = text.rfind(_BOXED_OPEN)
    if idx < 0:
        return None
    start = idx + len(_BOXED_OPEN)
    depth = 1
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


def extract_math_answer(text: str) -> str | None:
    """Try \\boxed{} first, then GSM8K '#### X', then 'the answer is X',
    then fall back to the last numeric token in the response.
    Returns None iff none of those match."""
    if text is None:
        return None
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed.strip()
    m = _HASHHASH.search(text)
    if m:
        return m.group(1).strip()
    m = _ANSWER_IS.search(text)
    if m:
        return m.group(1).strip().rstrip(".").strip()
    nums = _LAST_NUMBER.findall(text)
    if nums:
        return nums[-1]
    return None


def extract_code(text: str) -> str:
    """Return the LAST triple-backticked code block (`python` or unlabeled).

    If no fence is found, return the input unchanged — many models emit code
    without fences, and the caller (CE) will run whatever we give it.
    Empty / None input returns ''.
    """
    if not text:
        return ""
    matches = _CODE_FENCE.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def normalize(s: str | None) -> str:
    """Strip whitespace, $ and trailing periods, drop commas and spaces; collapse '72.0' -> '72'."""
    if s is None:
        return ""
    s = s.strip()
    s = s.strip("$").strip().rstrip(".").strip()
    s = s.replace(",", "").replace(" ", "")
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return repr(f)
    except (ValueError, OverflowError):
        return s


def parse_float_or_none(s: str | None) -> float | None:
    """Like float(s) but returns None on failure instead of raising."""
    if s is None:
        return None
    cleaned = str(s).strip().strip("$").replace(",", "").replace(" ", "").rstrip(".")
    try:
        return float(cleaned)
    except (ValueError, TypeError, OverflowError):
        return None
