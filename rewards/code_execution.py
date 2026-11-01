"""Code Execution (CE) reward: the study's execution scorer.

Math: the extracted answer is compared with the gold answer by SymPy equality at
      zero tolerance (numerical_tolerance._math_passes with atol=0).

Code: unit-test pass rate. Generated code runs in a subprocess with a timeout
      (10 s by default); the reward is the fraction of tests that pass.

Test formats, chosen by gold["source"]:
    (a) MBPP: gold_tests is a list of "assert ..." strings.
    (b) HumanEval: gold_tests is a single 'def check(candidate): ...' function
        containing one or more assert statements.
    (c) APPS: gold_tests[0] is a JSON string with either a "fn_name" (the
        candidate is called with each inputs[i] and compared with outputs[i])
        or plain stdin/stdout pairs.

Namespace separation: candidate code and the comparison never share a Python
namespace. Candidate code and a small fixed snippet that computes the value
under test run in a subprocess; a separate scope in that subprocess prints the
value's repr() behind a random marker; the parent process parses it with
ast.literal_eval and compares plain built-in values. A candidate therefore
cannot pass by returning an object with a custom __eq__ or __bool__.
Assertions that cannot be decomposed this way (about 1.2% of MBPP and 6.7% of
HumanEval problems in the evaluation set) are not run and score 0.0; see
_quarantined_unextractable_test.

This is not a security boundary. Subprocess isolation bounds run time but does
not sandbox the file system or the network, and candidate code shares a process
with the reporter, so it can inspect interpreter frames. Run untrusted code only
inside a sandbox.
"""
from __future__ import annotations

import ast
import json
import logging
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .base import Reward
from . import _extract

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Subprocess runner                                                            #
# --------------------------------------------------------------------------- #
# NOTE: a per-process resource cap (RLIMIT_* via preexec_fn) is deliberately absent.
# preexec_fn forces subprocess off posix_spawn() onto fork()+exec(), which is slow when
# forking from a large CUDA-initialised process, and forking from a multi-threaded parent
# risks deadlock. A correct cap needs another mechanism (a worker pool created before CUDA,
# or external cgroup limits). Keep scoring concurrency modest instead.
def _run_python_capture(code: str, timeout_s: float,
                         stdin_text: str | None = None) -> tuple[int | None, str, str]:
    """Run `code` in a subprocess, return (returncode, stdout, stderr).
    returncode is None on timeout or a launch/OS error."""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8",
        ) as fh:
            fh.write(code)
            tmp_path = fh.name
        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            return proc.returncode, proc.stdout or "", proc.stderr or ""
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
    except subprocess.TimeoutExpired:
        return None, "", f"timeout after {timeout_s}s"
    except (OSError, ValueError) as exc:
        return None, "", f"runner error: {exc}"




# --------------------------------------------------------------------------- #
# Namespace-separated candidate execution (see the module docstring)           #
# --------------------------------------------------------------------------- #
def _new_marker() -> str:
    """A fresh, cryptographically random, per-invocation marker. Never
    appears in candidate source or in any error text, so a candidate
    cannot forge a report by guessing/printing it -- this is what makes
    phase 2's stdout line trustworthy without needing to trust anything
    that ran in phase 1."""
    return "___RESULT_" + secrets.token_hex(16) + "___"


def _run_candidate_and_report(
    candidate_code: str,
    compute_code: str,
    timeout_s: float,
    stdin_text: str | None = None,
) -> str | None:
    """Run `candidate_code` and then `compute_code` (a fixed snippet that assigns
    the value under test to `_res`) in one subprocess namespace, then report
    repr(_res) to stdout behind a fresh random marker, from a separate scope that
    candidate code does not run in.

    Returns the text reported after the marker, or None if nothing was reported
    (crash, timeout, early exit, `_res` never assigned, or repr() raising); the
    caller treats None as a failure. Parsing and comparison happen in the parent
    process, which never runs candidate code. Candidate code can still inspect
    interpreter frames and overwrite the captured repr, so it could report a
    constant guess, but it never sees the expected value."""
    marker = _new_marker()
    script = (
        "import sys as _sys, builtins as _builtins\n"
        "_real_repr = _builtins.repr\n"
        "\n"
        # Convenience imports for the candidate -- plain stdlib names, not
        # protect-worthy trusted logic, so handing them to the candidate
        # (which can do whatever it wants with its OWN copy of them) is
        # harmless: nothing here helps a candidate hijack a comparison or
        # forge a report, unlike the _real_repr/marker machinery above and
        # below, which never appears in _candidate_globals.
        "import math as _math, itertools as _itertools, functools as _functools\n"
        "import collections as _collections, heapq as _heapq, bisect as _bisect\n"
        "import re as _re, string as _string, random as _random\n"
        "from typing import (List as _List, Tuple as _Tuple, Dict as _Dict,\n"
        "    Set as _Set, Optional as _Optional, Union as _Union, Any as _Any,\n"
        "    Iterable as _Iterable, Iterator as _Iterator, Callable as _Callable)\n"
        "from collections import (defaultdict as _defaultdict, deque as _deque,\n"
        "    Counter as _Counter, OrderedDict as _OrderedDict)\n"
        "_candidate_globals = {\n"
        "    '__name__': '__candidate__',\n"
        "    'math': _math, 'itertools': _itertools, 'functools': _functools,\n"
        "    'collections': _collections, 'heapq': _heapq, 'bisect': _bisect,\n"
        "    're': _re, 'string': _string, 'random': _random,\n"
        "    'List': _List, 'Tuple': _Tuple, 'Dict': _Dict, 'Set': _Set,\n"
        "    'Optional': _Optional, 'Union': _Union, 'Any': _Any,\n"
        "    'Iterable': _Iterable, 'Iterator': _Iterator, 'Callable': _Callable,\n"
        "    'defaultdict': _defaultdict, 'deque': _deque, 'Counter': _Counter,\n"
        "    'OrderedDict': _OrderedDict,\n"
        "}\n"
        f"_candidate_and_compute = {(candidate_code + chr(10) + chr(10) + compute_code)!r}\n"
        "try:\n"
        "    exec(_candidate_and_compute, _candidate_globals)\n"
        "except BaseException:\n"
        "    pass\n"
        "\n"
        "if '_res' in _candidate_globals:\n"
        "    try:\n"
        f"        _sys.stdout.write({marker!r} + _real_repr(_candidate_globals['_res']) + chr(10))\n"
        "        _sys.stdout.flush()\n"
        "    except BaseException:\n"
        "        pass\n"
    )
    _rc, out, _err = _run_python_capture(script, timeout_s, stdin_text=stdin_text)
    reported = None
    for line in out.splitlines():
        if line.startswith(marker):
            reported = line[len(marker):]
    return reported


def _literal_eval_safe(text: str) -> tuple[bool, Any]:
    """Best-effort ast.literal_eval. Returns (ok, value); ok is False if
    `text` isn't a genuine Python literal (covers both "candidate's repr
    was garbage/a default object repr" and "candidate's repr was
    malicious text that isn't valid literal syntax anyway" -- either way,
    literal_eval executes no code, so there's nothing to exploit even
    against fully adversarial input text)."""
    try:
        return True, ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False, None


def _values_match(actual: Any, expected: Any) -> bool:
    """Compare two plain values in the parent process. `actual` comes from
    ast.literal_eval of the reported text and `expected` from the dataset, so
    neither can be a candidate-defined object. Tuples and lists are normalized
    recursively (JSON has no tuple type), and a match against the sole element of
    a one-element expected list is accepted (the APPS convention). This unwrapping
    also applies to HumanEval and MBPP, where it can credit a scalar returned in
    place of a one-element list. Membership in a nested list is not accepted."""
    def norm(v: Any) -> Any:
        if isinstance(v, (tuple, list)):
            return [norm(x) for x in v]
        return v

    candidates = [expected]
    if isinstance(expected, list) and len(expected) == 1:
        candidates.append(expected[0])
    normed_actual = norm(actual)
    return any(normed_actual == norm(c) for c in candidates)






# --------------------------------------------------------------------------- #
# Code: unit-test pass rate                                                    #
# --------------------------------------------------------------------------- #
def _candidate_compiles(code: str) -> bool:
    """True iff `code` is syntactically valid Python (cheap pre-check)."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# Assertions that `_try_extract_from_assert_node` cannot decompose (about 1.2% of MBPP and
# 6.7% of HumanEval problems here) cannot be checked in the parent process, so they are
# not run and score 0.0, correct or not.
def _quarantined_unextractable_test(timeout_s: float) -> bool:
    """Return False (zero credit) for an assertion that cannot be decomposed. `timeout_s` is unused."""
    del timeout_s
    return False


_COMPARE_OP_TO_NAME = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Is: "is",
    ast.IsNot: "is not",
}


def _try_extract_from_assert_node(
    node: ast.Assert, source_text: str,
) -> tuple | None:
    """Try to extract a PARENT-SIDE-VERIFIABLE shape from `node`. Returns
    a tagged tuple, one of:
        ("always_true",)                            -- see below
        ("always_false",)                           -- see below
        ("truthy", call_source_text)                -- `assert <expr>`
        ("falsy", call_source_text)                 -- `assert not <expr>`
        ("compare", call_source_text, op_name, expected_value)
                                                      -- `assert <expr> <op> <literal>`
    or None if the shape isn't one we can extract (caller scores 0.0 for
    that assertion instead of running it unprotected -- see
    _quarantined_unextractable_test).

    Covers `assert <call> == <literal>` (the dominant shape, ~92% of all
    assert statements in the MBPP+HumanEval corpus),
    `assert True, "<msg>"`-style trivial placeholders (zero exploit
    surface -- no candidate-controlled value involved), and boolean
    truthy/falsy checks (`assert candidate(...)` / `assert not
    candidate(...)`, verified safe the same way as "compare": truthiness
    is checked in the parent process on an already ast.literal_eval'd
    value, which can't have a hijacked __bool__). Together these extract
    all but ~1.2% of MBPP and ~6.7% of HumanEval problems.

    Extraction is safe regardless of shape because it operates on
    TRUSTED, pre-authored dataset text (ast.literal_eval and
    ast.get_source_segment execute no code), never on candidate text; the
    extracted call source is evaluated later, inside the candidate's own
    namespace, by our own "compute" snippet (see
    _run_candidate_and_report) rather than as part of the original
    assert statement -- this is what moves the actual verification out of
    candidate-shared territory and into the trusted parent process.
    """
    test = node.test
    if isinstance(test, ast.Constant):
        try:
            return ("always_true",) if bool(test.value) else ("always_false",)
        except (TypeError, ValueError):
            return None
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        call_source = ast.get_source_segment(source_text, test.operand)
        if not call_source:
            return None
        return ("falsy", call_source)
    if isinstance(test, ast.Call):
        call_source = ast.get_source_segment(source_text, test)
        if not call_source:
            return None
        return ("truthy", call_source)
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op_name = _COMPARE_OP_TO_NAME.get(type(test.ops[0]))
        if op_name is not None:
            try:
                expected = ast.literal_eval(test.comparators[0])
            except (ValueError, SyntaxError, TypeError):
                return None
            left_source = ast.get_source_segment(source_text, test.left)
            if not left_source:
                return None
            return ("compare", left_source, op_name, expected)
    return None


def _try_extract_call_and_expected(test_str: str) -> tuple | None:
    """Single-statement variant of _try_extract_from_assert_node, for
    MBPP's one-assert-per-string convention -- parses `test_str` fresh
    and requires it to contain exactly one top-level Assert statement."""
    try:
        tree = ast.parse(test_str)
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return None
    return _try_extract_from_assert_node(tree.body[0], test_str)


# Batching a chunk's calls into one subprocess shares one wall-clock budget. A per-call
# watchdog (SIGALRM on Linux; a no-op on Windows, where the outer timeout is the only bound)
# limits EACH call to `timeout_s` seconds. The outer subprocess timeout is widened to fit the
# chunk (timeout_s * batch size = 100 s by default) and capped at 120 s, so a chunk in which
# every call hangs cannot stall scoring indefinitely.
_MAX_BATCH_TIMEOUT_S = 120.0

# After a call times out, the remaining calls of the same problem are marked failed without
# running: a structural hang (the same broken loop) tends to recur on the other calls, and
# waiting for each one dominated scoring time. Ordinary exceptions do not trigger this, so
# every remaining call still runs for its own partial credit.
#
# Names this helper defines live in the candidate's own namespace (`_candidate_globals`), so
# a fixed name could collide with a name legitimate candidate code uses (for example a
# module-level `_res`). Every name defined here therefore carries a fresh random per-call
# suffix. Only `_res`, the one rendezvous name the design needs, is unsuffixed.
#
# Known residual: the suffix prevents accidental collisions, not deliberate ones. A candidate
# could inspect globals() during the batch loop and overwrite the helpers, neutering the
# watchdog for later calls. That can make a candidate stall longer but cannot fake a correct
# score, because `expected` is never sent into the subprocess. Closing it would need the
# helpers captured as closure locals.
def _safe_call_helper(suffix: str) -> str:
    return (
        f"import signal as _signal_{suffix}\n"
        f"_HAS_ALARM_{suffix} = hasattr(_signal_{suffix}, 'SIGALRM')\n"
        f"class _CallTimeout_{suffix}(BaseException):\n"
        "    pass\n"
        f"def _alarm_handler_{suffix}(_signum, _frame):\n"
        f"    raise _CallTimeout_{suffix}()\n"
        f"if _HAS_ALARM_{suffix}:\n"
        f"    _signal_{suffix}.signal(_signal_{suffix}.SIGALRM, _alarm_handler_{suffix})\n"
        f"def _safe_{suffix}(_f, _per_call_s):\n"
        f"    if _HAS_ALARM_{suffix}:\n"
        f"        _signal_{suffix}.alarm(max(1, int(_per_call_s)))\n"
        "    try:\n"
        "        return (True, _f())\n"
        f"    except _CallTimeout_{suffix}:\n"
        "        return ('TIMEOUT', None)\n"
        "    except BaseException:\n"
        "        return (False, None)\n"
        "    finally:\n"
        f"        if _HAS_ALARM_{suffix}:\n"
        f"            _signal_{suffix}.alarm(0)\n"
    )


def _run_safe_batch(
    generated: str, preamble: str, call_sources: list[str], timeout_s: float,
) -> list[tuple[bool, Any]] | None:
    """Evaluate `call_sources` (Python expression source strings, each
    valid in `generated`'s namespace once `preamble` has run) in one subprocess call rather than one call per source, to avoid
    repeated interpreter start-up and temporary-file overhead.

    Each call is wrapped with `_safe()` so a raised exception on ONE call
    cannot silently discard the others: unlike _humaneval_pass_rate
    (where any failing assert legitimately fails the whole check() call,
    so a bare list literal is correct there), MBPP and APPS score partial
    credit per test -- a candidate that's right on 9 of 10 cases must
    still get credit for those 9, not 0 because the 10th raised.

    A genuine per-call TIMEOUT is different from a raised exception: once
    one call in the batch times out, the remaining calls are marked
    failed WITHOUT being run at all, rather than each independently
    waiting out its own timeout. A structural bug that hangs a candidate
    on one input overwhelmingly tends to hang it on the others too (same
    broken loop/condition, different data), so this trades a rare loss of
    partial credit (a candidate that hangs on exactly one input but would
    have passed later ones) for avoiding N sequential timeouts on what is
    almost always a candidate broken on everything.

    The outer subprocess timeout scales with batch size (`timeout_s` per
    call, capped at _MAX_BATCH_TIMEOUT_S) so legitimately-slow-but-
    correct multi-call batches aren't starved of the time they'd have
    had one-at-a-time; a candidate that hangs on every single call in the
    batch still can't exceed the cap regardless of how large the batch
    is (each hang is caught by its own per-call alarm well before the
    outer bound, but the cap is the backstop if that somehow fails, e.g.
    a candidate catching and swallowing the alarm's exception -- a known,
    accepted limitation of signal-based timeouts generally, not specific
    to this file).

    Returns a list of (ok, value) tuples, same length/order as
    `call_sources` (ok=False when that call raised or timed out), or
    None if nothing could be reported at all (crash, timeout, or a
    malformed report) -- callers must then treat every call in this
    batch as failed."""
    if not call_sources:
        return []
    # The loop's bookkeeping names would share the candidate's namespace, like `_res`, so each
    # carries a fresh random suffix (see _safe_call_helper) to avoid colliding with names that
    # legitimate candidate code uses. Only `_res` stays fixed.
    suffix = secrets.token_hex(6)
    call_thunks = [f"(lambda: ({c}))" for c in call_sources]
    compute_code = (
        _safe_call_helper(suffix) + preamble +
        f"def _run_batch_{suffix}():\n"
        "    _out = []\n"
        "    _hit_timeout = False\n"
        f"    for _thunk in [{', '.join(call_thunks)}]:\n"
        "        if _hit_timeout:\n"
        "            _out.append(('SKIPPED', None))\n"
        "            continue\n"
        f"        _r = _safe_{suffix}(_thunk, {timeout_s!r})\n"
        "        if _r[0] == 'TIMEOUT':\n"
        "            _hit_timeout = True\n"
        "        _out.append(_r)\n"
        "    return _out\n"
        f"_res = _run_batch_{suffix}()\n"
    )
    batch_timeout = min(timeout_s * len(call_sources), _MAX_BATCH_TIMEOUT_S)
    reported = _run_candidate_and_report(generated, compute_code, batch_timeout)
    if reported is None:
        return None
    lit_ok, actual_list = _literal_eval_safe(reported)
    if not lit_ok or not isinstance(actual_list, list) or len(actual_list) != len(call_sources):
        return None
    out: list[tuple[bool, Any]] = []
    for item in actual_list:
        # item[0] must be the literal `True` specifically -- 'TIMEOUT' and
        # 'SKIPPED' are truthy strings but must NOT count as success.
        if isinstance(item, tuple) and len(item) == 2 and item[0] is True:
            out.append((True, item[1]))
        else:
            out.append((False, None))
    return out


def _op_matches(op_name: str, actual: Any, expected: Any) -> bool:
    """Apply the extracted comparison operator using PARENT-SIDE,
    already-trusted values (see _values_match's docstring for why this is
    safe). `is`/`is not` are evaluated as value equality, because `actual`
    is a value rebuilt by ast.literal_eval in this process. This is looser
    than the original assertion: for example 1 is accepted where the test
    says `is True`."""
    if op_name == "==":
        return _values_match(actual, expected)
    if op_name == "!=":
        return not _values_match(actual, expected)
    if op_name == "is":
        return _values_match(actual, expected)
    if op_name == "is not":
        return not _values_match(actual, expected)
    return False


def _mbpp_pass_rate(generated: str, tests: list[str], timeout_s: float) -> float:
    """MBPP-style: each test is a standalone 'assert candidate(...) == ...'
    string.

    Tries Tier-1 extraction (see _try_extract_call_and_expected) on each
    test string. "always_true"/"always_false" entries resolve with no
    subprocess call at all; everything else is batched into ONE
    subprocess call for the whole problem via _run_safe_batch (MBPP
    problems have at most a handful of tests -- no chunking needed,
    unlike APPS). The comparison against each extracted expected value
    still happens entirely in this (trusted) process -- immune to a
    hijacked __eq__/__ne__ on any candidate-defined object, since the
    comparison never touches one. On extraction failure (an uncommon
    test shape -- see that function's docstring for exact frequencies,
    ~1.2% of MBPP problems), the assertion is scored 0.0 and not run
    -- see _quarantined_unextractable_test.
    """
    if not tests:
        return 0.0
    extracted_list = [_try_extract_call_and_expected(t or "") for t in tests]

    results = [False] * len(extracted_list)
    call_positions = []
    for i, extracted in enumerate(extracted_list):
        if extracted is None:
            results[i] = _quarantined_unextractable_test(timeout_s)
        elif extracted[0] == "always_true":
            results[i] = True
        elif extracted[0] == "always_false":
            results[i] = False
        else:
            call_positions.append(i)

    if call_positions:
        call_sources = [extracted_list[i][1] for i in call_positions]
        batch = _run_safe_batch(generated, "", call_sources, timeout_s)
        if batch is not None:
            for (call_ok, actual), idx in zip(batch, call_positions):
                if not call_ok:
                    continue
                entry = extracted_list[idx]
                kind = entry[0]
                if kind == "truthy":
                    results[idx] = bool(actual)
                elif kind == "falsy":
                    results[idx] = not bool(actual)
                else:  # "compare"
                    _tag, _call, op_name, expected = entry
                    results[idx] = _op_matches(op_name, actual, expected)

    return sum(results) / len(tests)


def _humaneval_pass_rate(generated: str, tests: list[str], entry_point: str | None,
                          timeout_s: float) -> float:
    """HumanEval-style: tests[0] is a single 'def check(candidate): ...'
    function containing one or more assert statements, ALL of which must
    pass for the whole test to count as a pass (matching the original
    check(candidate) semantics: it's one function call, any failing
    assert inside it raises and fails the whole thing).

    Without entry_point we cannot safely determine which of the
    candidate's functions to bind as `candidate`, so we score 0 rather
    than guessing.

    Tries Tier-1 extraction (_try_extract_from_assert_node) on EVERY
    assert statement found anywhere in tests[0] (via ast.walk, so it
    doesn't matter how deeply nested inside `check`'s body they are). If
    ALL of them extract cleanly, evaluates every extracted call
    expression in one namespace-separated pass (binding `candidate` to
    entry_point first) and compares each reported value against its own
    expected value entirely in this trusted process -- same protection
    as _mbpp_pass_rate, applied per-assert. If ANY assert doesn't extract
    cleanly (~6.7% of HumanEval problems), the whole problem is scored
    0.0 and not run -- see _quarantined_unextractable_test.
    """
    if not tests:
        return 0.0
    if not entry_point:
        logger.warning("humaneval record missing entry_point; scoring 0 "
                        "(cannot safely bind `candidate` for check())")
        return 0.0
    test_code = tests[0] or ""

    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return 0.0
    assert_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]

    extracted_all: list[tuple] | None = []
    if assert_nodes:
        for node in assert_nodes:
            extracted = _try_extract_from_assert_node(node, test_code)
            if extracted is None:
                extracted_all = None
                break
            extracted_all.append(extracted)
    else:
        extracted_all = None  # nothing to extract; fall back

    if extracted_all is not None:
        # "always_true"/"always_false" entries need no call at all;
        # everything else ("truthy"/"falsy"/"compare") has its call
        # source at index 1. Batch just the ones that need evaluating
        # into ONE list expression (one subprocess for the whole
        # problem, matching the original check(candidate) call count),
        # remembering which extracted-entry each batched position
        # corresponds to so results can be matched back up afterward.
        if any(e[0] == "always_false" for e in extracted_all):
            # A literal `assert False` anywhere makes the whole check()
            # fail unconditionally -- no candidate code needs to run at
            # all to know that (this shape is not expected in legitimate
            # benchmark data, but handling it costs nothing).
            return 0.0
        needs_call_indices = [
            i for i, e in enumerate(extracted_all) if e[0] != "always_true"
        ]
        if not needs_call_indices:
            # Every assert was a trivial always-true placeholder.
            return 1.0
        call_sources = [extracted_all[i][1] for i in needs_call_indices]
        compute_code = (
            f"candidate = {entry_point}\n"
            f"_res = [{', '.join(call_sources)}]\n"
        )
        reported = _run_candidate_and_report(generated, compute_code, timeout_s)
        if reported is not None:
            lit_ok, actual_list = _literal_eval_safe(reported)
            if lit_ok and isinstance(actual_list, list) and len(actual_list) == len(needs_call_indices):
                all_ok = True
                for actual, idx in zip(actual_list, needs_call_indices):
                    entry = extracted_all[idx]
                    kind = entry[0]
                    if kind == "truthy":
                        entry_ok = bool(actual)
                    elif kind == "falsy":
                        entry_ok = not bool(actual)
                    else:  # "compare"
                        _tag, _call, op_name, expected = entry
                        entry_ok = _op_matches(op_name, actual, expected)
                    if not entry_ok:
                        all_ok = False
                        break
                return 1.0 if all_ok else 0.0
        return 0.0

    return 1.0 if _quarantined_unextractable_test(timeout_s) else 0.0


def _intify_dict_keys(obj: Any) -> Any:
    """APPS problems with dict-valued inputs/outputs are authored with int
    keys, but JSON always serializes dict keys as strings. The official
    evaluator reconstructs the original int keys before comparing; without
    this, a candidate correctly using an int key (e.g. `d[1]`) against our
    JSON-decoded dict (which only has the string key "1") gets a spurious
    KeyError."""
    if isinstance(obj, dict):
        return {
            (int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k):
                _intify_dict_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_intify_dict_keys(x) for x in obj]
    return obj


# APPS function tests run in chunks of _APPS_FN_BATCH_SIZE per subprocess. A problem can
# have about 300 test cases: one subprocess for all of them would let a single hanging case
# zero the whole problem, and one per case would multiply interpreter start-up cost.
#
# Each chunk constructs a new `Solution()` instance, so state does not persist between
# chunks (the official APPS evaluator reuses one instance).
_APPS_FN_BATCH_SIZE = 10


def _apps_pass_rate(generated: str, io_json: str, timeout_s: float) -> float:
    """APPS-style: io_json is '{"inputs": [...], "outputs": [...], "fn_name":
    optional}'. fn_name present -> call the candidate per input and compare
    to the matching output; fn_name absent -> stdin/stdout: feed the input
    string on stdin, compare stdout to expected.

    fn_name-based problems are frequently posed as a `class Solution:`
    with the target as a method (the LeetCode-style convention), not a
    bare module-level function -- the compute snippet below tries
    Solution().{fn_name} first, falling back to the module-level function
    on any failure to resolve it (Solution undefined, lacks the method,
    or isn't instantiable). That resolution runs ONCE per chunk (not per
    test) since it's the same candidate code throughout.

    Test cases are evaluated in chunks of _APPS_FN_BATCH_SIZE via
    _run_safe_batch, one subprocess call per chunk instead of one per
    test (see _run_safe_batch's docstring and _APPS_FN_BATCH_SIZE's
    comment for why chunked rather than one giant batch or fully
    per-test). The actual comparison (does the candidate's return value
    match `expected`) happens entirely in THIS process via
    _values_match, using a value obtained through ast.literal_eval of
    text reported by the namespace-separated primitive -- see the module
    docstring (namespace separation).

    The stdin/stdout path (fn_name absent) launches one subprocess per test
    case.
    """
    try:
        io = json.loads(io_json or "")
    except (json.JSONDecodeError, TypeError):
        return 0.0
    inputs = _intify_dict_keys(io.get("inputs") or [])
    outputs = _intify_dict_keys(io.get("outputs") or [])
    if not inputs or not outputs or len(inputs) != len(outputs):
        return 0.0
    fn_name = io.get("fn_name")

    n_pass = 0
    if fn_name:
        preamble = (
            "try:\n"
            f"    _callable = getattr(Solution(), {fn_name!r})\n"
            "except (NameError, AttributeError, TypeError):\n"
            f"    _callable = globals()[{fn_name!r}]\n"
        )
        for start in range(0, len(inputs), _APPS_FN_BATCH_SIZE):
            chunk_inputs = inputs[start:start + _APPS_FN_BATCH_SIZE]
            chunk_expected = outputs[start:start + len(chunk_inputs)]
            call_sources = [
                f"_callable(*{(inp if isinstance(inp, list) else [inp])!r})"
                for inp in chunk_inputs
            ]
            batch = _run_safe_batch(generated, preamble, call_sources, timeout_s)
            if batch is None:
                continue
            for (call_ok, actual), expected in zip(batch, chunk_expected):
                if call_ok and _values_match(actual, expected):
                    n_pass += 1
    else:
        for inp, expected in zip(inputs, outputs):
            stdin_text = inp if isinstance(inp, str) else "\n".join(str(x) for x in inp)
            expected_text = expected if isinstance(expected, str) else "\n".join(str(x) for x in expected)
            # No guard needed on this path: correctness is fully
            # determined by comparing captured stdout to the expected
            # text (already happens entirely in this process, see
            # _run_stdin_python), so a legitimate solution calling
            # sys.exit(0) after printing its answer is fine, not a hack.
            ok, _ = _run_stdin_python(generated, stdin_text, expected_text, timeout_s)
            if ok:
                n_pass += 1
    return n_pass / len(inputs)


def _normalize_stdout(s: str) -> str:
    """Line-wise whitespace normalization for stdin/stdout comparison,
    matching ordinary competitive-programming judge leniency (and the
    official APPS evaluator): leading/trailing whitespace on each line and
    trailing blank lines don't affect correctness."""
    lines = s.replace("\r\n", "\n").split("\n")
    lines = [line.strip() for line in lines]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _run_stdin_python(code: str, stdin_text: str, expected_stdout: str,
                       timeout_s: float) -> tuple[bool, str]:
    """Run the candidate with `stdin_text` on stdin. Pass only if it exits cleanly
    and its normalized stdout equals `expected_stdout`. The comparison happens in
    this process on the captured output."""
    rc, out, err = _run_python_capture(code, timeout_s, stdin_text=stdin_text)
    if rc is None:
        return False, err
    if rc != 0:
        return False, err.strip()[:500]
    ok = _normalize_stdout(out) == _normalize_stdout(expected_stdout)
    return ok, "" if ok else f"stdout mismatch: {out.strip()[:200]!r}"


def _code_pass_rate(generated: str, tests: list[str], timeout_s: float,
                     source: str | None = None, entry_point: str | None = None) -> float:
    """Dispatch to the correct per-benchmark scoring path. See the module
    docstring for what each source does differently and why.

    A missing/unrecognized `source` falls through to MBPP-style scoring
    UNLESS the test text itself unambiguously looks like a HumanEval
    check() function (tests[0] starting with 'def check(') -- HumanEval's
    shape is unambiguous regardless of what the caller claims, and
    trusting the caller alone would let a caller bug silently mis-score
    HumanEval as MBPP.
    """
    if not tests:
        return 0.0
    if not _candidate_compiles(generated):
        return 0.0

    looks_like_humaneval = (tests[0] or "").lstrip().startswith("def check(")
    if source == "humaneval" or (source != "apps" and looks_like_humaneval):
        if source != "humaneval":
            logger.warning(
                "gold['source']=%r but tests[0] looks like a HumanEval "
                "check() function; scoring as HumanEval anyway instead of "
                "silently no-op'ing check(). Fix the caller to pass "
                "source='humaneval'.", source,
            )
        return _humaneval_pass_rate(generated, tests, entry_point, timeout_s)
    if source == "apps":
        return _apps_pass_rate(generated, tests[0] if tests else "", timeout_s)
    if source not in (None, "mbpp"):
        logger.warning("unrecognized gold['source']=%r; falling back to "
                        "MBPP-style scoring", source)
    return _mbpp_pass_rate(generated, tests, timeout_s)


# --------------------------------------------------------------------------- #
# Reward                                                                       #
# --------------------------------------------------------------------------- #
class CodeExecutionReward(Reward):
    def __init__(self, domain: str, timeout_s: float = 10.0):
        assert domain in ("math", "code"), domain
        self.domain = domain
        self.timeout_s = float(timeout_s)

    def compute(self, prompt: str, response: str, gold: dict) -> float:
        if self.domain == "math":
            gold_answer = (gold or {}).get("answer")
            if gold_answer is None:
                return 0.0
            extracted = _extract.extract_math_answer(response or "")
            if extracted is None:
                return 0.0
            # Math: SymPy equality at zero tolerance, using NT's comparator.
            from .numerical_tolerance import _math_passes
            return 1.0 if _math_passes(extracted, str(gold_answer), atol=0.0) else 0.0

        # Code domain
        tests = (gold or {}).get("tests") or []
        # Defensive: tests should be a list of strings.
        if isinstance(tests, str):
            tests = [tests]
        if not isinstance(tests, list):
            return 0.0
        generated = _extract.extract_code(response or "")
        if not generated:
            return 0.0
        source = (gold or {}).get("source")
        entry_point = (gold or {}).get("entry_point")
        return float(_code_pass_rate(generated, tests, self.timeout_s,
                                      source=source, entry_point=entry_point))
