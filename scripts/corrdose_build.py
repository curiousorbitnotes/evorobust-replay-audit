"""Build the sample files for the milder-corruption experiment.

Design: take the base model's second-run samples (Qwen-2.5-7B base, 421 prompts), the first N_SAMPLES (=32) samples of each
prompt. Write two files with identical prompt records:
  corrdose_clean32__code.samples.jsonl : the 32 original samples (re-judged, so the clean verdicts do not depend on the
                                         cache-pairing assumption)
  corrdose_brk32__code.samples.jsonl   : the same 32 samples, each with ONE surplus closing bracket ")" appended to the end of
                                         a randomly chosen code line inside the last fenced block (the block the study's
                                         extractor takes), chosen so that ast.parse of the extracted code fails.
Samples that already do not parse are left unchanged (they stay invalid). Samples where no line can be broken this way
(e.g. no executable lines outside strings) are left unchanged and counted.
Doses (10 / 30 / 60 percent) are then mixtures: a fixed uniform draw u per sample, corrupted iff u < dose, so doses are nested;
verdicts for clean and corrupted versions come from the two judged files (see corrdose_analyze.py).

Run from the repository root: python scripts/corrdose_build.py   (CPU, seconds)
Output: data/corrdose/*.samples.jsonl + data/corrdose/manifest.json (then judge the two samples files with the strong
judge and put the two .slj_scores.jsonl files next to them).
"""
import ast, json, os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rewards._extract import extract_code

SRC = os.path.join(os.environ.get('EVOROBUST_SAMPLES', os.path.join(ROOT, 'data', 'samples')),
                   'base_qwen7b_code__code.samples.jsonl')
OUTD = os.path.join(ROOT, 'data', 'corrdose')
N_SAMPLES = 32
rng = np.random.default_rng(20261012)
FENCE = re.compile(r'```[^\n]*\n(.*?)```', re.S)


def parses(code):
    if not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def corrupt(s):
    """append ')' to a random code line of the last fenced block so that the extracted code no longer parses; None if impossible"""
    ms = list(FENCE.finditer(s))
    if ms:
        m = ms[-1]; a, b = m.start(1), m.end(1)
    else:
        a, b = 0, len(s)
    body = s[a:b]
    lines = body.split('\n')
    cand = [i for i, l in enumerate(lines) if len(l.strip()) >= 3 and not l.strip().startswith('#')]
    for i in rng.permutation(cand):
        new = lines.copy(); new[i] = new[i] + ')'
        t = s[:a] + '\n'.join(new) + s[b:]
        if not parses(extract_code(t)):
            return t
    return None


os.makedirs(OUTD, exist_ok=True)
clean_recs, brk_recs = [], []
stats = dict(total=0, already_invalid=0, corrupted=0, could_not_corrupt=0)
corrupted_mask = []
for line in open(SRC, encoding='utf-8'):
    r = json.loads(line)
    smp = r['samples'][:N_SAMPLES]
    out, mask = [], []
    for s in smp:
        stats['total'] += 1
        if not parses(extract_code(s)):
            stats['already_invalid'] += 1; out.append(s); mask.append(0); continue
        t = corrupt(s)
        if t is None:
            stats['could_not_corrupt'] += 1; out.append(s); mask.append(0)
        else:
            stats['corrupted'] += 1; out.append(t); mask.append(1)
    base = {k: v for k, v in r.items() if k != 'samples'}
    clean_recs.append({**base, 'samples': smp})
    brk_recs.append({**base, 'samples': out})
    corrupted_mask.append(mask)

for name, recs in (('corrdose_clean32__code', clean_recs), ('corrdose_brk32__code', brk_recs)):
    with open(os.path.join(OUTD, name + '.samples.jsonl'), 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
u = rng.random((len(clean_recs), N_SAMPLES))
json.dump({'source_file': os.path.basename(SRC), 'n_prompts': len(clean_recs), 'n_samples': N_SAMPLES, 'stats': stats,
           'corrupted_mask': corrupted_mask, 'u': u.tolist(), 'doses': [0.1, 0.3, 0.6],
           'operator': 'append ")" to a random code line of the last fenced block, kept only if ast.parse(extract_code(.)) fails'},
          open(os.path.join(OUTD, 'manifest.json'), 'w'))
print(stats)
