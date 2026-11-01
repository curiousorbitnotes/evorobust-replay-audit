"""Rule out truncation as the cause of the replayed pass's parse failures.

If the k=128 pass simply hit a generation-length cap, the dominant error
would be an unclosed bracket at end of input, and failing samples would pile
up at the maximum length. An UNMATCHED CLOSING bracket is the opposite
signature: a token the model emitted that had no opener.
"""
import ast, json, os
from collections import Counter
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAY = os.path.join(os.environ.get('EVOROBUST_SAMPLES', os.path.join(ROOT, 'data', 'samples')))
import sys
sys.path.insert(0, ROOT)
from rewards._extract import extract_code  # the study's own extractor


for run in ("base_qwen7b_code__code", "slj_code_s1__code"):
    p = os.path.join(REPLAY, run + ".samples.jsonl")
    if not os.path.exists(p): continue
    L_ok, L_bad, kinds = [], [], Counter()
    for line in open(p, encoding="utf-8"):
        for s in json.loads(line)["samples"]:
            src = extract_code(s)
            n = len(s or "")
            if not src.strip():  # empty extracted code counts as invalid, as in the parse gate
                L_bad.append(n); kinds["EMPTY (no code extracted)"] += 1
                continue
            try:
                ast.parse(src); L_ok.append(n)
            except SyntaxError as e:
                L_bad.append(n)
                m = (e.msg or "").lower()
                if "was never closed" in m or "unexpected eof" in m:
                    kinds["UNCLOSED (truncation signature)"] += 1
                elif "unmatched" in m or "does not match" in m:
                    kinds["EXTRA CLOSER (emission signature)"] += 1
                else:
                    kinds["other"] += 1
            except Exception:
                L_bad.append(n); kinds["other"] += 1
    L_ok, L_bad = np.array(L_ok), np.array(L_bad)
    allL = np.concatenate([L_ok, L_bad]) if len(L_bad) else L_ok
    cap = allL.max()
    print("=" * 68)
    print(run)
    print("  chars: parsing  median %6.0f  p95 %6.0f  max %6.0f  n=%d"
          % (np.median(L_ok), np.percentile(L_ok, 95), L_ok.max(), len(L_ok)))
    if len(L_bad):
        print("  chars: failing  median %6.0f  p95 %6.0f  max %6.0f  n=%d"
              % (np.median(L_bad), np.percentile(L_bad, 95), L_bad.max(),
                 len(L_bad)))
        print("  failing samples within 2%% of the longest sample: %.2f%%"
              % (100 * (L_bad > 0.98 * cap).mean()))
        print("  parsing samples within 2%% of the longest sample: %.2f%%"
              % (100 * (L_ok > 0.98 * cap).mean()))
    tot = sum(kinds.values())
    for k, v in kinds.most_common():
        print("    %-36s %7d  %5.1f%%" % (k, v, 100 * v / max(tot, 1)))
