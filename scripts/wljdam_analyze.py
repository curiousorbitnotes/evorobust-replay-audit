"""Weak 7B judge on the damaged (second-run) checkpoints (CPU only).

Inputs : weak-judge verdicts wljdam_{ce,slj}_code_s1__code.wlj_scores.jsonl (WLJDAM_DIR, default data/wljdam),
         plus the second-run samples, strong-judge verdicts and hardened results read by evorobust_analyses.py.
Reports, per damaged run: weak-judge acceptance of parsing and non-parsing samples, weak-judge Pass@k (raw and gated
by the parse flag) at k = 1, 32, 128 with prompt-bootstrap intervals, and agreement with the strong judge. There are no
weak-judge verdicts for the base model's second-run samples, so no base-minus-damaged separation is computed for the
weak judge. Verdict j is assumed to belong to sample j.
Output : tables/wljdam_v1.json
Run    : python scripts/wljdam_analyze.py   (from the repository root)
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from evorobust_analyses import KS, RUNS, boot_mean_diff, load_run, parses, passk_vec

W = os.environ.get('WLJDAM_DIR', os.path.join(ROOT, 'data', 'wljdam'))
OUTJ = os.path.join(ROOT, 'tables', 'wljdam_v1.json')


def main():
    out = {'note': 'weak judge = Qwen2.5-7B-Instruct; verdict j is assumed to belong to sample j'}
    for lab, tag, canon in RUNS[1:]:
        d = load_run(tag, canon)
        wl = {}
        for l in open(os.path.join(W, f'wljdam_{tag}.wlj_scores.jsonl'), encoding='utf-8'):
            r = json.loads(l); assert r['prompt_idx'] not in wl; wl[r['prompt_idx']] = r['scores']
        idx = list(range(d['n'])); assert sorted(wl) == idx
        WJ = np.array([[1 if v >= 1 - 1e-9 else 0 for v in wl[i]] for i in idx])
        assert WJ.shape == d['J'].shape == (421, 128)
        P = np.array([[parses(s) for s in row] for row in d['resp']], dtype=int)
        res = {'weak_accept_rate_overall': float(WJ.mean()), 'strong_accept_rate_overall': float(d['J'].mean()),
               'weak_accept_nonparsing': float(WJ[P == 0].mean()), 'weak_accept_parsing': float(WJ[P == 1].mean()),
               'strong_accept_nonparsing': float(d['J'][P == 0].mean()), 'strong_accept_parsing': float(d['J'][P == 1].mean()),
               'n_nonparsing': int((P == 0).sum()), 'n_parsing': int((P == 1).sum()),
               'weak_strong_agreement': float((WJ == d['J']).mean()), 'weak_strong_agreement_nonparsing': float((WJ[P == 0] == d['J'][P == 0]).mean()),
               'passk': {}}
        for k in KS:
            v_raw = passk_vec(WJ, k); v_gate = passk_vec(WJ * P, k); v_s = passk_vec(d['J'], k); v_ex = passk_vec(d['H'], k)
            res['passk'][f'k={k}'] = {'weak_judge': {'mean': float(v_raw.mean()), 'ci95': boot_mean_diff(v_raw)},
                                      'weak_judge+gate': {'mean': float(v_gate.mean()), 'ci95': boot_mean_diff(v_gate)},
                                      'strong_judge': float(v_s.mean()), 'execution_hardened': float(v_ex.mean()),
                                      'weak_minus_exec': {'mean': float((v_raw - v_ex).mean()), 'ci95': boot_mean_diff(v_raw, v_ex)}}
        out[lab] = res
        print(lab, {k: ('%.2f%%' % (100 * v) if isinstance(v, float) else v) for k, v in res.items() if k != 'passk'})
        for k, v in res['passk'].items():
            print('  ', k, 'weak %.2f%%' % (100 * v['weak_judge']['mean']),
                  'weak + gate %.2f%%' % (100 * v['weak_judge+gate']['mean']),
                  'strong %.2f%%' % (100 * v['strong_judge']),
                  'exec %.2f%%' % (100 * v['execution_hardened']))
    os.makedirs(os.path.dirname(OUTJ), exist_ok=True)
    json.dump(out, open(OUTJ, 'w'), indent=1)
    print('WROTE', OUTJ)


if __name__ == '__main__':
    main()
