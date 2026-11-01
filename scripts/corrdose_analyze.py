"""Milder-corruption dose-response. Run after the strong judge has scored the two files built by corrdose_build.py
(corrdose_clean32__code.slj_scores.jsonl and corrdose_brk32__code.slj_scores.jsonl, read from JUDGE_DIR, default
data/corrdose).

For each dose d in {0, 0.1, 0.3, 0.6, 1.0}: sample (i,j) is corrupted iff it was corruptible (manifest mask) and u[i][j] < d.
 judge     : verdict of the corrupted file if corrupted else verdict of the (re-judged) clean file
 judge+gate: judge AND extracted code parses
 execution : hardened strict_ok of the clean sample if not corrupted, 0 if corrupted (the corrupted code does not parse,
             so no harness can pass it; checked below)
Pass@k with n = 32 samples per prompt at k = 1, 8, 32, prompt-bootstrap 95% intervals (5,000 draws).
Also reported: the judge's acceptance rate on corrupted samples vs the same samples clean, and agreement of the re-judged clean
verdicts with the cached verdicts (a side check of the cache pairing for the first 32 samples).
Output: tables/corrdose_v1.json
"""
import ast, json, os, sys
from math import comb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rewards._extract import extract_code

JD = os.environ.get('JUDGE_DIR', os.path.join(ROOT, 'data', 'corrdose'))
DD = os.path.join(ROOT, 'data', 'corrdose')
AUD = os.path.join(ROOT, 'data', 'hardened_scores')
REPLAY = os.path.join(os.environ.get('EVOROBUST_SAMPLES', os.path.join(ROOT, 'data', 'samples')))
OUT = os.path.join(ROOT, 'tables', 'corrdose_v1.json')
KS = (1, 8, 32)
NB = 5000
rng = np.random.default_rng(3)


def pak(c, n, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def load_scores(path):
    d = {}
    for l in open(path, encoding='utf-8'):
        r = json.loads(l); d[r['prompt_idx']] = r['scores']
    return np.array([[1 if v >= 1 - 1e-9 else 0 for v in d[i]] for i in sorted(d)])


def parses(s):
    c = extract_code(s)
    if not c.strip():
        return False
    try:
        ast.parse(c); return True
    except Exception:
        return False


man = json.load(open(os.path.join(DD, 'manifest.json')))
mask = np.array(man['corrupted_mask'], bool); u = np.array(man['u']); N = man['n_samples']
JC = load_scores(os.path.join(JD, 'corrdose_clean32__code.slj_scores.jsonl'))
JB = load_scores(os.path.join(JD, 'corrdose_brk32__code.slj_scores.jsonl'))
assert JC.shape == JB.shape == mask.shape == (421, N), (JC.shape, JB.shape, mask.shape)
clean = [json.loads(l) for l in open(os.path.join(DD, 'corrdose_clean32__code.samples.jsonl'), encoding='utf-8')]
brk = [json.loads(l) for l in open(os.path.join(DD, 'corrdose_brk32__code.samples.jsonl'), encoding='utf-8')]
PC = np.array([[parses(s) for s in r['samples']] for r in clean], int)
PB = np.array([[parses(s) for s in r['samples']] for r in brk], int)
assert (PB[mask] == 0).all(), 'a "corrupted" sample still parses'
H = np.array(json.load(open(os.path.join(AUD, 'base_qwen7b_code__code.canon4_sample_level.json')))['strict_ok'], int)[:, :N]
cached = load_scores(os.path.join(REPLAY, 'base_qwen7b_code__code.slj_scores.jsonl'))[:, :N]

res = {'n_prompts': 421, 'n_samples': N, 'corruptible_share': float(mask.mean()), 'manifest_stats': man['stats'],
       'side_checks': {'rejudged_clean_vs_cached_agreement': float((JC == cached).mean()),
                       'cached_accept_rate': float(cached.mean()), 'rejudged_clean_accept_rate': float(JC.mean())},
       'judge_on_corrupted_samples': {'accept_rate_corrupted': float(JB[mask].mean()), 'accept_rate_same_samples_clean': float(JC[mask].mean()),
                                      'n': int(mask.sum())}, 'doses': {}}
src = np.array([r['source'] for r in clean])
unchanged = np.array([[a == b for a, b in zip(rc['samples'], rb['samples'])] for rc, rb in zip(clean, brk)])
res['side_checks']['repeatability_unchanged_samples'] = {'n': int(unchanged.sum()),
                                                         'verdict_agreement_clean_vs_second_call': float((JC == JB)[unchanged].mean())}
idx_b = rng.integers(0, 421, size=(NB, 421))  # one set of prompt resamples for all paired intervals


def ci_mean(v):
    b = np.sort(v[idx_b].mean(1)); return [float(b[125]), float(b[4874])]


vec = {}
for d in (0.0, 0.1, 0.3, 0.6, 1.0):
    cor = mask & (u < d) if d < 1.0 else mask.copy()
    J = np.where(cor, JB, JC)
    P = np.where(cor, PB, PC)
    E = np.where(cor, 0, H)
    inst = {'judge': J, 'judge+gate': J * P, 'execution_hardened': E}
    out = {'corrupted_share_of_samples': float(cor.mean()), 'parse_rate': float(P.mean())}
    for k in KS:
        out[f'k={k}'] = {}
        for name, M in inst.items():
            v = np.array([pak(int(r.sum()), N, k) for r in M])
            vec[(d, k, name)] = v
            out[f'k={k}'][name] = {'mean': float(v.mean()), 'ci95': ci_mean(v)}
        # judge minus execution at this dose (paired over prompts) and per benchmark source
        dv = vec[(d, k, 'judge')] - vec[(d, k, 'execution_hardened')]
        out[f'k={k}']['judge_minus_execution'] = {'mean': float(dv.mean()), 'ci95': ci_mean(dv)}
        out[f'k={k}']['by_source_means'] = {s: {n: float(vec[(d, k, n)][src == s].mean()) for n in inst} for s in ('humaneval', 'mbpp')}
    res['doses'][str(d)] = out
    print('corruption dose %.0f%%' % (100 * d),
          {k: '%.2f%%' % (100 * out[f'k={k}']['judge']['mean']) for k in KS},
          'parse %.2f%%' % (100 * out['parse_rate']))
# paired change from dose 0 (prompt-paired)
res['paired_change_from_dose_0'] = {}
for d in (0.1, 0.3, 0.6, 1.0):
    res['paired_change_from_dose_0'][str(d)] = {}
    for k in KS:
        res['paired_change_from_dose_0'][str(d)][f'k={k}'] = {}
        for name in ('judge', 'judge+gate', 'execution_hardened'):
            dv = vec[(d, k, name)] - vec[(0.0, k, name)]
            res['paired_change_from_dose_0'][str(d)][f'k={k}'][name] = {'mean': float(dv.mean()), 'ci95': ci_mean(dv)}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, 'w'), indent=1)
print('WROTE', OUT)
