"""LLM-as-Judge reward (shared class for SLJ and WLJ).

Sends a single-turn chat prompt to a vLLM-served Qwen judge:
    - SLJ: Qwen-2.5-72B-Instruct on http://localhost:8000/v1
    - WLJ: Qwen-2.5-7B-Instruct  on http://localhost:8001/v1

Returns a binary 0.0 / 1.0: a leading verdict word or digit is read first, then the
first 0 or 1 anywhere in the reply.
The vLLM server must be running; this module does not start it.

The scoring prompt template lives here as `JUDGE_PROMPT_TEMPLATE` and is
the prompt used by every judge experiment. The same prompt is used for the
strong and the weak judge.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from .base import Reward

logger = logging.getLogger(__name__)


JUDGE_PROMPT_TEMPLATE = """\
You are an expert grader. Your job is to decide whether a candidate
answer is CORRECT or INCORRECT for the given problem.

Output exactly one character: '1' if the candidate answer is correct,
or '0' if it is incorrect. Do not output anything else — no reasoning,
no punctuation, no whitespace.

PROBLEM:
{prompt}

GOLD ANSWER:
{gold}

CANDIDATE ANSWER:
{response}

YOUR VERDICT (1 or 0):"""


_BINARY_RE = re.compile(r"[01]")


# Module-level cached tokenizer. Qwen2.5-7B and Qwen2.5-72B share the same
# tokenizer (Qwen2.5 family), so we load the smaller 7B variant once and
# reuse it for both SLJ and WLJ. Tokenizer files are ~50MB and load in <1s.
_QWEN_TOKENIZER: Any = None


def _get_qwen_tokenizer() -> Any:
    global _QWEN_TOKENIZER
    if _QWEN_TOKENIZER is False:  # sentinel for "tried and failed"
        return None
    if _QWEN_TOKENIZER is not None:
        return _QWEN_TOKENIZER
    try:
        from transformers import AutoTokenizer
        _QWEN_TOKENIZER = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct"
        )
        return _QWEN_TOKENIZER
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "could not load Qwen tokenizer (%s); falling back to char-based "
            "truncation. Token counts will be approximate.", exc,
        )
        _QWEN_TOKENIZER = False
        return None


def _parse_binary(text: str) -> float:
    """Map judge text -> 0.0 or 1.0. Best-effort:
        '1' / '1.\\n' / 'correct' / 'yes' -> 1.0
        '0' / 'incorrect' / 'no'         -> 0.0
        anything ambiguous               -> 0.0 (treat as incorrect)
    """
    if not text:
        return 0.0
    head = text.strip().lower()
    if head.startswith("1") or head.startswith("yes") or head.startswith("correct"):
        return 1.0
    if head.startswith("0") or head.startswith("no") or head.startswith("incorrect"):
        return 0.0
    m = _BINARY_RE.search(head)
    if m:
        return 1.0 if m.group(0) == "1" else 0.0
    return 0.0


class LLMJudgeReward(Reward):
    """LLM-as-judge reward.

    Constructor takes the judge model, vLLM endpoint, and decoding params.
    `compute()` issues a single chat-completion request per (prompt, response,
    gold) tuple. Network failures or non-2xx responses return 0.0 to keep
    training going (with a warning); operators should monitor for an
    elevated network-failure rate during runs.
    """

    def __init__(
        self,
        judge_model: str,
        domain: str,
        vllm_endpoint: str = "http://localhost:8000/v1",
        temperature: float = 0.0,
        timeout_s: float = 30.0,
        max_tokens: int = 4,
        max_input_chars: int | None = None,
        max_input_tokens: int | None = None,
    ):
        assert domain in ("math", "code"), domain
        self.judge_model = judge_model
        self.domain = domain
        # Strip trailing slash so /chat/completions concatenates cleanly.
        self.vllm_endpoint = vllm_endpoint.rstrip("/")
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.max_tokens = int(max_tokens)
        # Char budget (legacy, char-based fallback).
        if max_input_chars is None:
            max_input_chars = int(os.environ.get("JUDGE_MAX_INPUT_CHARS", "10000"))
        self.max_input_chars = int(max_input_chars)
        # Token budget (preferred, tokenizer-accurate). Defaults pick a safe
        # value based on the judge model: 7B → 4096 max_len → 3500 token
        # budget; 72B → 8192 max_len → 7500 token budget. Override with
        # JUDGE_MAX_INPUT_TOKENS env var or constructor arg.
        if max_input_tokens is None:
            env_val = os.environ.get("JUDGE_MAX_INPUT_TOKENS")
            if env_val is not None:
                max_input_tokens = int(env_val)
            elif "72B" in judge_model.upper():
                max_input_tokens = 7500  # SLJ Qwen-72B at max_len=8192
            else:
                max_input_tokens = 3500  # WLJ Qwen-7B at max_len=4096
        self.max_input_tokens = int(max_input_tokens)
        # Eagerly load tokenizer so we know whether token-based truncation
        # is available. _get_qwen_tokenizer() caches at module level.
        self._tokenizer = _get_qwen_tokenizer()

    # ------------------------------------------------------------------ #
    # HTTP                                                                #
    # ------------------------------------------------------------------ #
    def _build_messages(self, prompt: str, response: str, gold: dict) -> list[dict]:
        gold_str = ""
        if gold:
            gold_str = str(gold.get("answer") or "")
            tests = gold.get("tests") or []
            if tests and not gold_str:
                gold_str = "Tests that must pass:\n" + "\n".join(str(t) for t in tests)

        prompt = prompt or ""
        response = response or ""

        if self._tokenizer is not None:
            prompt, response, gold_str = self._truncate_by_tokens(
                prompt, response, gold_str,
            )
        else:
            prompt, response, gold_str = self._truncate_by_chars(
                prompt, response, gold_str,
            )

        body = JUDGE_PROMPT_TEMPLATE.format(
            prompt=prompt,
            gold=gold_str,
            response=response,
        )
        return [{"role": "user", "content": body}]

    def _truncate_by_tokens(
        self, prompt: str, response: str, gold_str: str,
    ) -> tuple[str, str, str]:
        """Tokenizer-accurate truncation. Guarantees the assembled body fits
        within self.max_input_tokens by shrinking components in priority
        order (problem-statement first, then response, then gold last).

        We re-encode after each truncation pass to avoid drift from token
        boundary effects. Total worst-case latency: ~3-5ms per call.
        """
        tok = self._tokenizer
        # Reserve headroom for the chat template wrapping (~30-50 tokens for
        # Qwen2.5: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...
        # <|im_end|>\n<|im_start|>assistant\n) plus output tokens.
        chat_overhead = 60
        budget = self.max_input_tokens - chat_overhead - self.max_tokens
        if budget < 200:
            budget = 200

        # Quick path: encode once, check whether truncation is needed at all.
        body = JUDGE_PROMPT_TEMPLATE.format(
            prompt=prompt, gold=gold_str, response=response,
        )
        body_tokens = tok.encode(body, add_special_tokens=False)
        if len(body_tokens) <= budget:
            return prompt, response, gold_str

        # Truncate the problem statement first.
        # Compute how many tokens we have to give up.
        # Encode the non-prompt scaffold (template + gold + response) to find
        # how much budget remains for the prompt.
        scaffold = JUDGE_PROMPT_TEMPLATE.format(
            prompt="", gold=gold_str, response=response,
        )
        scaffold_tokens = len(tok.encode(scaffold, add_special_tokens=False))
        prompt_budget = max(50, budget - scaffold_tokens)

        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > prompt_budget:
            # Reserve a few tokens for a [TRUNCATED] marker.
            keep = max(50, prompt_budget - 6)
            prompt_ids = prompt_ids[:keep]
            prompt = tok.decode(prompt_ids, skip_special_tokens=True) + "\n[TRUNCATED]"

        # Re-check; if still over budget, truncate the response.
        body = JUDGE_PROMPT_TEMPLATE.format(
            prompt=prompt, gold=gold_str, response=response,
        )
        if len(tok.encode(body, add_special_tokens=False)) <= budget:
            return prompt, response, gold_str

        # Truncate response: keep head + tail so we see both the
        # function signature and the boxed/return value.
        scaffold_no_resp = JUDGE_PROMPT_TEMPLATE.format(
            prompt=prompt, gold=gold_str, response="",
        )
        no_resp_tokens = len(tok.encode(scaffold_no_resp, add_special_tokens=False))
        resp_budget = max(50, budget - no_resp_tokens)

        resp_ids = tok.encode(response, add_special_tokens=False)
        if len(resp_ids) > resp_budget:
            head = resp_budget // 2
            tail = resp_budget - head - 6  # 6 tokens for marker
            tail = max(0, tail)
            response = (
                tok.decode(resp_ids[:head], skip_special_tokens=True)
                + "\n...[TRUNCATED]...\n"
                + tok.decode(resp_ids[-tail:], skip_special_tokens=True)
                if tail > 0
                else tok.decode(resp_ids[:head], skip_special_tokens=True) + "\n[TRUNCATED]"
            )

        # Re-check; if still over budget, truncate the gold answer. The caller wraps what this
        # function returns in the template once, so the budget is applied to the three fields only,
        # never to the already-templated message.
        body = JUDGE_PROMPT_TEMPLATE.format(
            prompt=prompt, gold=gold_str, response=response,
        )
        if len(tok.encode(body, add_special_tokens=False)) <= budget:
            return prompt, response, gold_str

        scaffold_no_gold = JUDGE_PROMPT_TEMPLATE.format(
            prompt=prompt, gold="", response=response,
        )
        no_gold_tokens = len(tok.encode(scaffold_no_gold, add_special_tokens=False))
        gold_budget = max(20, budget - no_gold_tokens)

        gold_ids = tok.encode(gold_str, add_special_tokens=False)
        if len(gold_ids) > gold_budget:
            keep = max(20, gold_budget - 6)
            gold_ids = gold_ids[:keep]
            gold_str = tok.decode(gold_ids, skip_special_tokens=True) + "\n[TRUNCATED]"

        # Final guarantee: verify the actual assembled body against
        # budget (decode->encode round-trips can shift token counts
        # slightly at truncation boundaries, so a single truncation pass
        # is not provably exact) and, if still over, shrink gold_str
        # directly in a bounded loop -- never re-wrap an
        # already-templated string in the template again.
        for _ in range(5):
            body = JUDGE_PROMPT_TEMPLATE.format(
                prompt=prompt, gold=gold_str, response=response,
            )
            body_tokens = tok.encode(body, add_special_tokens=False)
            if len(body_tokens) <= budget:
                break
            over_by = len(body_tokens) - budget
            gold_ids = tok.encode(gold_str, add_special_tokens=False)
            if len(gold_ids) <= 10:
                # Nothing left to safely cut from gold; give up on gold
                # and shrink response instead, which still has slack from
                # its own earlier truncation-to-budget pass.
                resp_ids = tok.encode(response, add_special_tokens=False)
                resp_ids = resp_ids[: max(10, len(resp_ids) - over_by - 10)]
                response = tok.decode(resp_ids, skip_special_tokens=True) + "\n[TRUNCATED]"
                continue
            gold_ids = gold_ids[: max(10, len(gold_ids) - over_by - 10)]
            gold_str = tok.decode(gold_ids, skip_special_tokens=True) + "\n[TRUNCATED]"

        return prompt, response, gold_str

    def _truncate_by_chars(
        self, prompt: str, response: str, gold_str: str,
    ) -> tuple[str, str, str]:
        """Char-based fallback used only when the tokenizer fails to load."""
        template_overhead = len(JUDGE_PROMPT_TEMPLATE)
        non_prompt_chars = template_overhead + len(gold_str) + len(response)
        prompt_budget = self.max_input_chars - non_prompt_chars
        if prompt_budget < 200:
            prompt_budget = 200
            resp_budget = self.max_input_chars - template_overhead - len(gold_str) - prompt_budget
            if resp_budget > 0 and len(response) > resp_budget:
                head = resp_budget // 2
                tail = resp_budget - head
                response = response[:head] + "\n...[TRUNCATED]...\n" + response[-tail:]
            elif resp_budget <= 0:
                gold_str = gold_str[: max(100, self.max_input_chars - template_overhead - 300)]
        if len(prompt) > prompt_budget:
            prompt = prompt[:prompt_budget] + "\n...[TRUNCATED]"
        return prompt, response, gold_str

    def _post(self, messages: list[dict]) -> str:
        """POST to the vLLM endpoint and return the model's text. Raises
        on non-2xx; caller catches and treats as 0.0."""
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests not installed") from exc

        payload = {
            "model": self.judge_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        url = f"{self.vllm_endpoint}/chat/completions"
        resp = requests.post(url, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        # OpenAI-compatible shape: choices[0].message.content
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected vLLM response shape: {data!r}") from exc

    # ------------------------------------------------------------------ #
    # Reward                                                              #
    # ------------------------------------------------------------------ #
    def compute(self, prompt: str, response: str, gold: dict) -> float:
        messages = self._build_messages(prompt, response, gold or {})
        try:
            text = self._post(messages)
        except Exception as exc:  # noqa: BLE001
            import os as _os
            if _os.environ.get("JUDGE_STRICT") == "1":
                # Offline scoring only (offline scoring should retry instead of caching a 0.0).
                # Training keeps the lenient default above so a flaky endpoint cannot crash a run.
                raise
            logger.warning(
                "LLMJudgeReward(%s) request failed: %s. Returning 0.0.",
                self.judge_model, exc,
            )
            return 0.0
        return _parse_binary(text)
