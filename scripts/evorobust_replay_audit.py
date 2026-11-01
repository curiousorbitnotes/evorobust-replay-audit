"""Replay audit: parse validity of two generation passes, judge acceptance and the parse gate.

Setting. A GRPO study archived k=32 code generations for every checkpoint (first run). Later, the same
checkpoints were downloaded again and re-generated at k=128 on different hardware with a different inference
stack (second run). The two passes are compared on whether the emitted programs are syntactically valid
Python at all.

No execution harness is needed here. A program that does not parse cannot run, so a positive judge verdict on
one is a false positive by definition. That makes `ast.parse` a necessary-condition gate: free, deterministic
and independent of the judge.

Output: tables/replay_audit.json (validity rates, the parse-error taxonomy, judge acceptance, gate intervals).
Run from the repository root:  python scripts/evorobust_replay_audit.py
"""
from __future__ import annotations

import ast
import json
import os
import re
from collections import Counter
from math import comb

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = os.environ.get('EVOROBUST_ARCHIVE', os.path.join(ROOT, 'data', 'archive_k32'))
REPLAY = os.path.join(os.environ.get('EVOROBUST_SAMPLES', os.path.join(ROOT, 'data', 'samples')))
import sys
sys.path.insert(0, ROOT)
from rewards._extract import extract_code  # the study's own extractor
OUT = os.path.join(ROOT, "tables")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260908)
NBOOT = 5000

# (label, archived file, replayed file, replayed judge-score file)
RUNS = [
    ("base Qwen-2.5-7B", "base_qwen7b__code.samples.jsonl",
     "base_qwen7b_code__code.samples.jsonl",
     "base_qwen7b_code__code.slj_scores.jsonl"),
    ("CE-trained", "ce_code_s1__code.samples.jsonl",
     "ce_code_s1__code.samples.jsonl",
     "ce_code_s1__code.slj_scores.jsonl"),
    ("SLJ-trained", "slj_code_s1__code.samples.jsonl",
     "slj_code_s1__code.samples.jsonl",
     "slj_code_s1__code.slj_scores.jsonl"),
]


# --------------------------------------------------------------- primitives


def parse_status(text):
    """Return (ok, short reason). Reason is '' when it parses."""
    src = extract_code(text)
    if not src.strip():
        return False, "empty"
    try:
        ast.parse(src)
        return True, ""
    except SyntaxError as e:
        msg = (e.msg or "syntax error").strip()
        # collapse the variable parts so the taxonomy has few buckets
        msg = re.sub(r"'[^']*'", "'...'", msg)
        return False, msg[:58]
    except Exception as e:
        return False, type(e).__name__


def taxonomy_rows(reasons):
    """Group parser messages into the rows of the paper's taxonomy table (f-string forms included)."""
    rows = {"unmatched closing bracket": 0, "closing bracket does not match opener": 0,
            "invalid syntax (unclassified)": 0, "positional argument follows keyword": 0,
            "other classes (incl. empty)": 0}
    for msg, n in reasons.items():
        if "unmatched" in msg:
            rows["unmatched closing bracket"] += n
        elif "does not match opening" in msg:
            rows["closing bracket does not match opener"] += n
        elif msg == "invalid syntax":
            rows["invalid syntax (unclassified)"] += n
        elif msg.startswith("positional argument follows keyword"):
            rows["positional argument follows keyword"] += n
        else:
            rows["other classes (incl. empty)"] += n
    return rows


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def pass_at_k(c, n, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def boot_ci(v, nboot=NBOOT):
    b = np.array([v[RNG.integers(0, len(v), len(v))].mean()
                  for _ in range(nboot)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


res = {"k_archived": None, "k_replay": None, "runs": {}}
print("=" * 74)
print("1. SYNTACTIC VALIDITY UNDER TWO INDEPENDENT GENERATION PASSES")
print("%-18s %-9s %7s %8s %9s %9s" %
      ("checkpoint", "pass", "k", "samples", "invalid", "HEval/MBPP"))

taxonomy = {}
for label, af, rf, _sf in RUNS:
    entry = {}
    for tag, path in (("archived", os.path.join(ARCH, af)),
                      ("replay", os.path.join(REPLAY, rf))):
        if not os.path.exists(path):
            print("  [missing] %s %s" % (label, tag))
            continue
        recs = load(path)
        k = len(recs[0]["samples"])
        tot = bad = 0
        by_src = {}
        reasons = Counter()
        for r in recs:
            src = r.get("source", "unknown")
            d = by_src.setdefault(src, [0, 0])
            for s in r["samples"]:
                ok, why = parse_status(s)
                tot += 1
                d[1] += 1
                if not ok:
                    bad += 1
                    d[0] += 1
                    reasons[why] += 1
        rate = bad / tot
        per_src = {s: v[0] / v[1] for s, v in sorted(by_src.items())}
        entry[tag] = {"k": k, "n": tot, "invalid": bad, "rate": rate,
                      "by_source": per_src}
        if tag == "replay":
            taxonomy[label] = reasons.most_common(6)
            entry["taxonomy_rows"] = taxonomy_rows(reasons)
            entry["n_distinct_parser_messages"] = len([m for m in reasons if m != "empty"])
            entry["n_empty_outputs"] = reasons.get("empty", 0)
        print("%-18s %-9s %7d %8d %8.2f%%  %s" %
              (label, tag, k, tot, 100 * rate,
               " / ".join("%.1f%%" % (100 * v) for v in per_src.values())))
        res["k_%s" % tag] = k
    res["runs"][label] = entry

print()
print("Most common parse failures in the replayed pass:")
for label, common in taxonomy.items():
    print("  %s" % label)
    for why, n in common:
        print("      %-58s %7d" % (why, n))

# ------------------------------------------------------------------------
print()
print("=" * 74)
print("2. DOES THE STRONG JUDGE NOTICE? (replayed pass, Qwen-2.5-72B judge)")
print("%-18s %9s %9s %10s %10s" %
      ("checkpoint", "invalid", "accepted", "accept-rate", "%of pos"))

for label, _af, rf, sf in RUNS:
    gp, sp = os.path.join(REPLAY, rf), os.path.join(REPLAY, sf)
    if not (os.path.exists(gp) and os.path.exists(sp)):
        continue
    scores = {r["prompt_idx"]: r["scores"] for r in load(sp)}
    n_inv = n_inv_pos = n_pos = n_val = n_val_pos = 0
    for r in load(gp):
        sc = scores.get(r["prompt_idx"])
        if sc is None:
            continue
        for s, v in zip(r["samples"], sc):
            pos = v >= 1.0 - 1e-9
            ok, _ = parse_status(s)
            n_pos += pos
            if ok:
                n_val += 1
                n_val_pos += pos
            else:
                n_inv += 1
                n_inv_pos += pos
    d = {"n_invalid": n_inv, "accepted_invalid": n_inv_pos,
         "accept_rate_invalid": n_inv_pos / max(n_inv, 1),
         "n_valid": n_val, "accepted_valid": n_val_pos,
         "accept_rate_valid": n_val_pos / max(n_val, 1),
         "n_positive": n_pos,
         "frac_positives_invalid": n_inv_pos / max(n_pos, 1)}
    res["runs"][label]["judge"] = d
    print("%-18s %9d %9d %9.2f%% %9.2f%%" %
          (label, n_inv, n_inv_pos, 100 * d["accept_rate_invalid"],
           100 * d["frac_positives_invalid"]))
    print("%-18s   (valid: %d accepted of %d = %.2f%%)" %
          ("", n_val_pos, n_val, 100 * d["accept_rate_valid"]))

# --------- two-judge control: base Llama has both a strong and a weak judge
print()
print("Optional historical two-judge control (base Llama-3.1-8B; not reported in the current paper):")
gp = os.path.join(REPLAY, "base_llama8b__code.samples.jsonl")
if os.path.exists(gp):
    recs = load(gp)
    ctrl = {}
    for jname, jf in (("strong", "base_llama8b__code.slj_scores.jsonl"),
                      ("weak", "base_llama8b__code.wlj_scores.jsonl")):
        sp = os.path.join(REPLAY, jf)
        if not os.path.exists(sp):
            continue
        scores = {r["prompt_idx"]: r["scores"] for r in load(sp)}
        n_inv = n_inv_pos = n_pos = 0
        for r in recs:
            sc = scores.get(r["prompt_idx"])
            if sc is None:
                continue
            for s, v in zip(r["samples"], sc):
                pos = v >= 1.0 - 1e-9
                n_pos += pos
                if not parse_status(s)[0]:
                    n_inv += 1
                    n_inv_pos += pos
        ctrl[jname] = {"n_invalid": n_inv, "accepted_invalid": n_inv_pos,
                       "accept_rate_invalid": n_inv_pos / max(n_inv, 1),
                       "frac_positives_invalid": n_inv_pos / max(n_pos, 1)}
        print("  %-6s judge: %d invalid, %d accepted (%.2f%%), "
              "%.2f%% of positives" %
              (jname, n_inv, n_inv_pos, 100 * ctrl[jname]["accept_rate_invalid"],
               100 * ctrl[jname]["frac_positives_invalid"]))
    res["llama_two_judge"] = ctrl

# ------------------------------------------------------------------------
print()
print("=" * 74)
print("3. THE GATE: judge verdict AND parses")
print("%-18s %9s %9s %10s %-22s" %
      ("checkpoint", "Pass@1", "gated", "drop (pp)", "95% CI of drop (pp)"))

gate = {}
for label, _af, rf, sf in RUNS:
    gp, sp = os.path.join(REPLAY, rf), os.path.join(REPLAY, sf)
    if not (os.path.exists(gp) and os.path.exists(sp)):
        continue
    scores = {r["prompt_idx"]: r["scores"] for r in load(sp)}
    u1, g1, uc, gc, nn = [], [], [], [], None
    for r in load(gp):
        sc = scores.get(r["prompt_idx"])
        if sc is None:
            continue
        nn = len(sc)
        cu = sum(1 for v in sc if v >= 1.0 - 1e-9)
        cg = sum(1 for s, v in zip(r["samples"], sc)
                 if v >= 1.0 - 1e-9 and parse_status(s)[0])
        u1.append(cu / nn)
        g1.append(cg / nn)
        uc.append(cu)
        gc.append(cg)
    u1, g1 = np.array(u1), np.array(g1)
    d = g1 - u1
    lo, hi = boot_ci(d)
    row = {"pass1": float(u1.mean()), "pass1_gated": float(g1.mean()),
           "delta": float(d.mean()), "ci": [lo, hi], "n_prompts": len(u1),
           "k": nn}
    for k in (8, 32, 128):
        if nn >= k:
            row["pass%d" % k] = float(np.mean(
                [pass_at_k(c, nn, k) for c in uc]))
            row["pass%d_gated" % k] = float(np.mean(
                [pass_at_k(c, nn, k) for c in gc]))
    gate[label] = row
    res["runs"][label]["gate"] = row
    print("%-18s %8.2f%% %8.2f%% %10.2f  [%.2f, %.2f]" %
          (label, 100 * row["pass1"], 100 * row["pass1_gated"],
           -100 * row["delta"], -100 * hi, -100 * lo))

print()
print("%-18s %8s %8s %8s %8s" % ("checkpoint", "P@32", "gated", "P@128",
                                 "gated"))
for label, row in gate.items():
    print("%-18s %7.2f%% %7.2f%% %7.2f%% %7.2f%%" %
          (label, 100 * row.get("pass32", float("nan")),
           100 * row.get("pass32_gated", float("nan")),
           100 * row.get("pass128", float("nan")),
           100 * row.get("pass128_gated", float("nan"))))

# ------------------------------------------------------------------------
print()
print("=" * 74)
print("4. THE TRIPWIRE: would the audit have fired without knowing the truth?")
base_rate = res["runs"]["base Qwen-2.5-7B"]["replay"]["rate"]
arch_rate = res["runs"]["base Qwen-2.5-7B"]["archived"]["rate"]
print("  reference validity (archived pass, any checkpoint): %.2f%% invalid"
      % (100 * arch_rate))
print("  a threshold anywhere in 1%%-50%% separates the two passes:")
for label in res["runs"]:
    a = res["runs"][label].get("archived", {}).get("rate")
    r = res["runs"][label].get("replay", {}).get("rate")
    if a is None or r is None:
        continue
    print("    %-18s archived %6.2f%%  replay %6.2f%%   %s"
          % (label, 100 * a, 100 * r,
             "FIRES" if r > 0.5 else "passes"))
res["tripwire_reference_rate"] = arch_rate
res["base_replay_rate"] = base_rate

with open(os.path.join(OUT, "replay_audit.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
print("\nwrote tables/replay_audit.json")
