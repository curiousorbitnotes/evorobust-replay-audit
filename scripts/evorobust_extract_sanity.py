"""Is the code-extraction step manufacturing the parse failures?

Three independent renderings of the same sample are parsed. If all three
agree that the replayed trained checkpoint emits unparseable Python, the
result cannot be an artefact of how the block is pulled out of the response.
"""
import ast, json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAY = os.path.join(os.environ.get('EVOROBUST_SAMPLES', os.path.join(ROOT, 'data', 'samples')))
import sys
sys.path.insert(0, ROOT)
from rewards._extract import extract_code  # the study's own extractor (last fenced block)

def ok(src):
    if not (src or "").strip(): return False
    try: ast.parse(src); return True
    except Exception: return False

def extractor_block(t):
    return extract_code(t)

def fenced_any(t):
    """Most permissive: parse succeeds if ANY fenced block parses, else raw."""
    ms = re.findall(r"```(?:python|py)?\s*\n(.*?)```", t or "", re.S | re.I)
    return ms if ms else [t or ""]

for run in ("base_qwen7b_code__code", "slj_code_s1__code"):
    p = os.path.join(REPLAY, run + ".samples.jsonl")
    if not os.path.exists(p): continue
    n = raw_ok = extractor_ok = any_ok = 0
    for line in open(p, encoding="utf-8"):
        for s in json.loads(line)["samples"]:
            n += 1
            raw_ok += ok(s)
            extractor_ok += ok(extractor_block(s))
            any_ok += any(ok(b) for b in fenced_any(s))
    print("%-24s n=%d  raw %.2f%%  extractor-block %.2f%%  ANY-block %.2f%% parse OK"
          % (run, n, 100*raw_ok/n, 100*extractor_ok/n, 100*any_ok/n))

print()
print("=" * 70)
print("VERBATIM TAIL OF THREE FAILING SAMPLES (slj_code_s1 replay)")
p = os.path.join(REPLAY, "slj_code_s1__code.samples.jsonl")
shown = 0
for line in open(p, encoding="utf-8"):
    for s in json.loads(line)["samples"]:
        if shown >= 3: break
        b = extractor_block(s)
        if ok(b): continue
        msg = "empty"
        try: ast.parse(b)
        except SyntaxError as e: msg = e.msg
        except Exception as e: msg = repr(e)
        print("-" * 70)
        print("[%s]  len(response)=%d  len(block)=%d" % (msg, len(s), len(b)))
        print(repr(b[-220:]))
        shown += 1
    if shown >= 3: break
