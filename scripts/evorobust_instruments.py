"""What each instrument reports about the SAME replayed generations.

Four readings per checkpoint at k=128: the fraction of samples that parse,
the judge's Pass@k, the judge's Pass@k gated on parsing, and the code
execution verifier's Pass@k (the study's cached scorer, ce_FINAL arrays). Also checks the necessary condition empirically:
execution should never award credit to a program that does not parse.
"""
import ast, json, os
from math import comb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAY = os.path.join(os.environ.get('EVOROBUST_SAMPLES', os.path.join(ROOT, 'data', 'samples')))
PPS = os.path.join(ROOT, "data", "per_prompt_scores")
OUT = os.path.join(ROOT, "tables")
import sys; sys.path.insert(0, ROOT)
from rewards._extract import extract_code

RUNS = [("base Qwen-2.5-7B", "base_qwen7b_code__code"),
        ("CE-trained", "ce_code_s1__code"),
        ("SLJ-trained", "slj_code_s1__code")]

def ok(t):
    s = extract_code(t)
    if not s.strip(): return False
    try: ast.parse(s); return True
    except Exception: return False

def pak(c, n, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)

rows, checks = [], []
for label, run in RUNS:
    recs = [json.loads(l) for l in
            open(os.path.join(REPLAY, run + ".samples.jsonl"), encoding="utf-8")]
    js = {r["prompt_idx"]: r["scores"] for r in
          [json.loads(l) for l in
           open(os.path.join(REPLAY, run + ".slj_scores.jsonl"), encoding="utf-8")]}
    ex = np.load(os.path.join(PPS, run + "__code128__ce_FINAL.npy"))
    n = ex.shape[1]
    cj, cg, ce, npar, ntot = [], [], [], 0, 0
    exec_pos_on_invalid = 0
    for i, r in enumerate(recs):
        sc = js.get(r["prompt_idx"])
        good = [ok(s) for s in r["samples"]]
        npar += sum(good); ntot += len(good)
        cj.append(sum(1 for v in sc if v >= 1 - 1e-9))
        cg.append(sum(1 for g, v in zip(good, sc) if v >= 1 - 1e-9 and g))
        ce.append(int((ex[i] >= 1.0 - 1e-9).sum()))
        exec_pos_on_invalid += int(sum(1 for g, e in zip(good, ex[i])
                                       if (not g) and e >= 1.0 - 1e-9))
    row = {"run": label, "parse_rate": npar / ntot,
           "exec_credit_on_nonparsing": exec_pos_on_invalid}
    for k in (1, 32, 128):
        row["judge@%d" % k] = float(np.mean([pak(c, n, k) for c in cj]))
        row["gated@%d" % k] = float(np.mean([pak(c, n, k) for c in cg]))
        row["exec@%d" % k] = float(np.mean([pak(c, n, k) for c in ce]))
    rows.append(row)
    checks.append((label, exec_pos_on_invalid, ntot - npar))

print("=" * 88)
print("WHAT EACH INSTRUMENT SAYS ABOUT THE SAME REPLAYED GENERATIONS (k=128)")
print("Levels are percentages; execution uses the study's cached scorer, not the hardened harness.")
print("%-18s %7s | %7s %7s %7s | %7s %7s %7s" %
      ("checkpoint", "parses", "J@1", "J@32", "J@128",
       "exec@1", "exec@32", "exec@128"))
for r in rows:
    print("%-18s %6.2f%% | %6.2f%% %6.2f%% %6.2f%% | %6.2f%% %6.2f%% %6.2f%%" %
          (r["run"], 100 * r["parse_rate"], 100 * r["judge@1"], 100 * r["judge@32"],
           100 * r["judge@128"], 100 * r["exec@1"], 100 * r["exec@32"], 100 * r["exec@128"]))
print()
print("%-18s %8s %8s %8s   (judge verdict AND parses)" %
      ("checkpoint", "gate@1", "gate@32", "gate@128"))
for r in rows:
    print("%-18s %7.2f%% %7.2f%% %7.2f%%" %
          (r["run"], 100 * r["gated@1"], 100 * r["gated@32"], 100 * r["gated@128"]))

print()
print("NECESSARY-CONDITION CHECK: execution credit awarded to non-parsing code")
for label, bad, ninv in checks:
    print("  %-18s %d of %d non-parsing samples scored 1.0 by execution"
          % (label, bad, ninv))

json.dump(rows, open(os.path.join(OUT, "instruments.json"), "w"), indent=2)
print("\nwrote tables/instruments.json")
