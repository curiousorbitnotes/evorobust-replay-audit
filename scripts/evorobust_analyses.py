"""Paired comparisons, simple-signal baselines and prose-versus-acceptance (CPU only).

1. Paired per-problem comparison with prompt-level bootstrap intervals: judge, judge + parse gate, execution
   (the study's cached scorer) and execution (hardened harness) at k = 1, 32, 128 on the second-run outputs,
   and the base-minus-damaged separation per instrument.
2. Simple signals against the parse gate: how well output length and code share tell damaged-run samples
   from base-run samples, compared with the parse flag (sample-level AUC, ties count one half).
3. Text outside the code versus judge acceptance, for parsing and non-parsing samples.

Inputs : second-run samples and strong-judge verdicts (EVOROBUST_SAMPLES), the study's cached execution
         scores (data/per_prompt_scores/*__code128__ce_FINAL.npy) and the hardened per-sample results
         (data/hardened_scores/*_sample_level.json). Judge verdict j is assumed to belong to sample j
         (see scripts/judge_recheck.py).
Output : tables/paired_analyses.json
Run    : python scripts/evorobust_analyses.py   (from the repository root)
"""
import ast, json, os, re, sys
from math import comb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAY = os.path.join(os.environ.get('EVOROBUST_SAMPLES', os.path.join(ROOT, 'data', 'samples')))
PPS = os.path.join(ROOT, 'data', 'per_prompt_scores')
AUD = os.path.join(ROOT, 'data', 'hardened_scores')
OUT = os.path.join(ROOT, 'tables', 'paired_analyses.json')
sys.path.insert(0, ROOT)
from rewards._extract import extract_code

RUNS = [('base', 'base_qwen7b_code__code', 'canon4'), ('CE-trained', 'ce_code_s1__code', 'canon5'),
        ('SLJ-trained', 'slj_code_s1__code', 'canon5')]
KS = (1, 32, 128)
FENCE = re.compile(r'```.*?```', re.S)
NB = 5000
PROSE_CHARS = 40
rng = np.random.default_rng(1)


def pak(c, n, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def parses(s):
    c = extract_code(s)
    if not c.strip():
        return False
    try:
        ast.parse(c)
        return True
    except Exception:
        return False


def load_run(tag, canon):
    recs = [json.loads(l) for l in open(os.path.join(REPLAY, tag + '.samples.jsonl'), encoding='utf-8')]
    idx = [r['prompt_idx'] for r in recs]
    assert idx == list(range(len(recs))), 'prompt order'
    js = {}
    for l in open(os.path.join(REPLAY, tag + '.slj_scores.jsonl'), encoding='utf-8'):
        r = json.loads(l); js[r['prompt_idx']] = r['scores']
    J = np.array([[1 if v >= 1 - 1e-9 else 0 for v in js[i]] for i in idx])
    E = (np.load(os.path.join(PPS, tag + '__code128__ce_FINAL.npy')) >= 1 - 1e-9).astype(int)
    sl = json.load(open(os.path.join(AUD, f'{tag}.{canon}_sample_level.json')))
    assert sl['prompt_idxs'] == idx
    H = np.array(sl['strict_ok'], dtype=int)
    assert len(js) == len(idx), 'duplicate or missing judge rows'
    assert J.shape == E.shape == H.shape == (len(idx), 128), (J.shape, E.shape, H.shape)
    resp = [r['samples'] for r in recs]
    src = [r['source'] for r in recs]
    return dict(J=J, E=E, H=H, resp=resp, src=src, n=len(recs))


def passk_vec(M, k):
    n = M.shape[1]
    return np.array([pak(int(r.sum()), n, k) for r in M])


def ci(boot):
    b = np.sort(boot); lo = int(0.025 * len(b)); hi = int(0.975 * len(b)) - 1
    return [float(b[lo]), float(b[hi])]


def boot_mean_diff(a, b=None):
    n = len(a)
    idx = rng.integers(0, n, size=(NB, n))
    v = a[idx].mean(1) if b is None else (a - b)[idx].mean(1)
    return ci(v)


def auc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie), via average ranks."""
    x = np.concatenate([pos, neg]); order = x.argsort(kind='mergesort'); ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x) + 1)
    xs = x[order]; i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]: j += 1
        if j > i: ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    data = {lab: load_run(tag, canon) for lab, tag, canon in RUNS}
    res = {'note': 'judge verdict j is assumed to belong to sample j; exec_study = the study\'s cached scorer '
                   '(ce_FINAL arrays); exec_hardened = strict_ok from canon4.py / canon5.py'}

    # parse flags and features
    for lab, d in data.items():
        P = np.array([[parses(s) for s in row] for row in d['resp']], dtype=int)
        d['P'] = P
        d['len'] = np.array([[len(s) for s in row] for row in d['resp']], dtype=float)
        d['code_share'] = np.array([[(len(extract_code(s)) / max(len(s), 1)) for s in row] for row in d['resp']], dtype=float)
        print(lab, 'parse rate', '%.2f%%' % (100 * P.mean()), flush=True)

    # ---- 1. Pass@k per instrument with prompt-level bootstrap, and paired base-minus-damaged separations ----
    passk = {}
    for lab, d in data.items():
        inst = {'judge': d['J'], 'judge+gate': d['J'] * d['P'], 'exec_study': d['E'], 'exec_hardened': d['H']}
        passk[lab] = {}
        for k in KS:
            vecs = {name: passk_vec(M, k) for name, M in inst.items()}
            d.setdefault('vecs', {})[k] = vecs
            passk[lab][f'k={k}'] = {name: {'mean': float(v.mean()), 'ci95': boot_mean_diff(v)} for name, v in vecs.items()}
            passk[lab][f'k={k}']['judge_minus_exec_hardened'] = {'mean': float((vecs['judge'] - vecs['exec_hardened']).mean()),
                                                                 'ci95': boot_mean_diff(vecs['judge'], vecs['exec_hardened'])}
            passk[lab][f'k={k}']['judge_minus_exec_study'] = {'mean': float((vecs['judge'] - vecs['exec_study']).mean()),
                                                             'ci95': boot_mean_diff(vecs['judge'], vecs['exec_study'])}
    sep = {}
    for dam in ('CE-trained', 'SLJ-trained'):
        sep[dam] = {}
        for k in KS:
            sep[dam][f'k={k}'] = {}
            for name in ('judge', 'judge+gate', 'exec_study', 'exec_hardened'):
                a = data['base']['vecs'][k][name]; b = data[dam]['vecs'][k][name]
                sep[dam][f'k={k}'][name] = {'base_minus_damaged': float((a - b).mean()), 'ci95_paired_prompts': boot_mean_diff(a, b)}
    res['pass_at_k_with_prompt_bootstrap'] = passk
    res['base_minus_damaged_separation_paired'] = sep
    print('Pass@k and separations done', flush=True)

    # ---- 2. Sample-level AUC for telling damaged-run samples from base-run samples ----
    signals = {}
    b = data['base']
    for dam in ('CE-trained', 'SLJ-trained'):
        d = data[dam]
        sub = lambda M: M.reshape(-1)
        signals[dam] = {}
        for fname, base_val, dam_val in (('parse_fail_flag', 1 - sub(b['P']), 1 - sub(d['P'])),
                                         ('output_length_chars', sub(b['len']), sub(d['len'])),
                                         ('low_code_share', 1 - sub(b['code_share']), 1 - sub(d['code_share']))):
            # AUC for ranking a damaged-run sample above a base-run sample (20,000 samples per side)
            rs = np.random.default_rng(2)
            i1 = rs.choice(len(dam_val), 20000, replace=False); i2 = rs.choice(len(base_val), 20000, replace=False)
            signals[dam][fname] = {'auc_damaged_vs_base': auc(dam_val[i1], base_val[i2]),
                                   'also_reversed_for_length': 1 - auc(dam_val[i1], base_val[i2]) if fname == 'output_length_chars' else None,
                                   'mean_base': float(base_val.mean()), 'mean_damaged': float(dam_val.mean())}
        # within the damaged run: does length predict non-parsing?
        nonp = d['len'][d['P'] == 0]; par = d['len'][d['P'] == 1]
        signals[dam]['within_run_length_auc_nonparsing_vs_parsing'] = auc(rs.choice(nonp, min(20000, len(nonp)), replace=False),
                                                                          rs.choice(par, min(20000, len(par)), replace=False))
    res['simple_signals_vs_parse_gate'] = signals
    print('Simple signals done', flush=True)

    # ---- 3. Text outside the code vs judge acceptance ----
    prose_res = {}
    for lab, d in data.items():
        prose = np.zeros_like(d['P'])
        for i, row in enumerate(d['resp']):
            for j, s in enumerate(row):
                # characters outside all fenced code blocks
                if '```' in s:
                    outside = len(FENCE.sub('', s).strip())
                else:
                    outside = len(s.strip()) - len(extract_code(s))
                prose[i, j] = int(outside > PROSE_CHARS)
        d['prose'] = prose
        prose_res[lab] = {}
        for pn, pm in (('no_prose', prose == 0), ('prose', prose == 1)):
            for qn, qm in (('parses', d['P'] == 1), ('non_parsing', d['P'] == 0)):
                m = pm & qm
                n = int(m.sum())
                if n == 0:
                    prose_res[lab][f'{pn}|{qn}'] = {'n': 0}; continue
                num = (d['J'] * m).sum(1).astype(float); den = m.sum(1).astype(float)
                rate = float(num.sum() / den.sum())
                keep = den > 0
                ix = rng.integers(0, keep.sum(), size=(NB, keep.sum()))
                nn, dd = num[keep][ix].sum(1), den[keep][ix].sum(1)
                prose_res[lab][f'{pn}|{qn}'] = {'n': n, 'n_prompts_contributing': int(keep.sum()), 'judge_accept_rate': rate,
                                                'ci95_prompt_cluster_conditional_on_contributing_prompts': ci(nn / dd)}
    res['prose_definition'] = f'more than {PROSE_CHARS} characters outside all fenced code blocks'
    res['prose_vs_judge_acceptance'] = prose_res
    print('Prose versus acceptance done', flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, 'w'), indent=1)
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
