"""Hardened execution harness for HumanEval/MBPP.

- a sample is credited only when a random success token is written after the tests have run; the token is
  written through references captured BEFORE candidate code runs (os.write, no json at result time), so a
  candidate that patches json.dumps / sys.stdout or exits early cannot forge a pass. It is still NOT an
  adversarial security boundary: do not use it as the only evaluator for reward-optimized checkpoints.
- HumanEval: the full problem prompt is executed first (helpers + imports defined by the prompt, as in the
  official prompt+completion assembly); the candidate then redefines the entry point.
- name adaptation is decided at RUN time: only for strict failures that were NameError on the expected target,
  target absent from the namespace, exactly one function defined by the candidate, no top-level class,
  every test call shape binds to that function's real signature (inspect.signature.bind). Alias only.
- controls abort the run if any reference solution fails or any negative control passes.
- per-sample outcomes (strict/adapted, error type, missing symbol) are saved; input/script/output hashes recorded;
  sample-level rates get prompt-cluster bootstrap CIs; NameError symbols are tabulated.
"""
import ast, builtins, hashlib, json, math, os, multiprocessing as mp, platform, random, secrets, sys
REPO = os.environ.get('EVOROBUST_REPO', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL = os.environ.get('EVOROBUST_SAMPLES', f'{REPO}/data/samples')
HARD = f'{REPO}/data/hardened_scores'
TABLES = f'{REPO}/tables'
os.makedirs(HARD, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)
OUT = f'{TABLES}/bootstrap_ci_pass128_v6_{os.environ.get("CANON_TAG", "base")}.json'
sys.path.insert(0, REPO)
from rewards import _extract
from rewards.code_execution import _run_python_capture

K = 128
N_BOOT = 10000
SEED = 1
_ENV_RUNS = os.environ.get('CANON_RUNS')
RUNS = ([(t, t) for t in [x.strip() for x in _ENV_RUNS.split(',')] if t] if _ENV_RUNS
        else [('base_qwen7b_code__code', 'Qwen-2.5-7B-Instruct (before GRPO), code'), ('base_llama8b__code', 'Llama-3.1-8B-Instruct, code')])
if len({t for t, _ in RUNS}) != len(RUNS) or not RUNS:
    raise SystemExit('CANON_RUNS empty or duplicated')
MBPP_PRE = ('import math, re, string, itertools, functools, collections, heapq, bisect, operator, sys\n'
            'from collections import *\nfrom itertools import *\nfrom functools import *\nfrom heapq import *\n')
BUILTINS = set(dir(builtins))
EPS = {r['prompt']: r['entry_point'] for r in json.load(open(f'{REPO}/data/code/eval.json'))
       if r.get('source') == 'humaneval' and r.get('entry_point')}

TEMPLATE = '''
import json as _json, os as _os, sys as _sys, types as _types, inspect as _inspect
_write = _os.write
_FT = _types.FunctionType
_isinst = isinstance
_sig = _inspect.signature
class _AliasBlocked(Exception):
    pass
def _clean(s):
    return ''.join(c for c in (s or '') if c.isalnum() or c == '_')[:60]
def _run():
    g = {'__name__': '__candidate__'}
    try:
        exec(compile(%(pre)r, '<pre>', 'exec'), g)
        exec(compile(%(cand)r, '<cand>', 'exec'), g)
        _alias = %(alias)r
        if _alias:
            _tgt, _shapes = _alias
            if _tgt in g:
                raise _AliasBlocked()
            _fs = [v for v in list(g.values()) if _isinst(v, _FT) and getattr(v, '__module__', None) == '__candidate__']
            _cl = [v for v in list(g.values()) if _isinst(v, type) and getattr(v, '__module__', None) == '__candidate__']
            if len(_fs) != 1 or _cl:
                raise _AliasBlocked()
            _s = _sig(_fs[0])
            for _n, _kws in _shapes:
                try:
                    _s.bind(*([0] * _n), **{k: 0 for k in _kws})
                except TypeError:
                    raise _AliasBlocked()
            g[_tgt] = _fs[0]
        exec(compile(%(tests)r, '<tests>', 'exec'), g)
        _entry = %(entry)r
        if _entry:
            if _entry not in g:
                raise NameError("entry point missing", name=_entry)
            g['check'](g[_entry])
        return (True, '', '')
    except BaseException as e:
        return (False, _clean(type(e).__name__), _clean(getattr(e, 'name', None) if _isinst(getattr(e, 'name', None), str) else ''))
_ok, _err, _sym = _run()
_msg = '\\n' + %(token)r + ('1' if _ok else '0') + '|' + _err + '|' + _sym + '\\n'
_write(1, _msg.encode())
'''


def norm_code(s):
    """extract_code + formatting-only fix: drop a leading standalone language-tag line
    ('python', 'py', 'python3') that appears when the opening code fence is missing."""
    code = _extract.extract_code(s or '')
    if code:
        first, _, rest = code.partition(chr(10))
        if first.strip().lower() in ('python', 'py', 'python3') and rest.strip():
            return rest
    return code


def fhash(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def pre_for(rec):
    return rec['prompt'] + '\n' if rec['source'] == 'humaneval' else MBPP_PRE


def run_one(args):
    key, pre, cand, tests, entry, alias, timeout = args
    token = '@@R' + secrets.token_hex(12) + '@@'
    prog = TEMPLATE % dict(pre=pre, cand=cand, tests=tests, entry=entry, alias=alias, token=token)
    rc, out, err = _run_python_capture(prog, timeout)
    if rc is None:
        if 'timeout' in (err or '').lower():
            return key, (False, 'Timeout', '')
        raise RuntimeError(f'runner launch failure for {key}: {err!r}')
    for line in reversed((out or '').splitlines()):
        if line.startswith(token):
            ok, e, s = (line[len(token):].split('|') + ['', '', ''])[:3]
            return key, (ok == '1', e, s)
    return key, (False, 'NoResult', '')


def target_and_shapes(tests_list):
    bound, calls = set(), {}
    for t in tests_list:
        try:
            tree = ast.parse(t)
        except SyntaxError:
            return None, None
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                bound.add(n.name)
            elif isinstance(n, ast.Import):
                for a in n.names: bound.add((a.asname or a.name).split('.')[0])
            elif isinstance(n, ast.ImportFrom):
                for a in n.names: bound.add(a.asname or a.name)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                bound.add(n.id)
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                calls.setdefault(n.func.id, []).append(n)
    names = {k for k in calls if k not in BUILTINS and k not in bound}
    if len(names) != 1:
        return None, None
    nm = next(iter(names))
    shapes = []
    for c in calls[nm]:
        if any(isinstance(a, ast.Starred) for a in c.args) or any(k.arg is None for k in c.keywords):
            return None, None
        shapes.append((len(c.args), [k.arg for k in c.keywords]))
    return nm, shapes


def load(tag):
    recs, seen = [], set()
    for line in open(f'{EVAL}/{tag}.samples.jsonl', encoding='utf-8'):
        r = json.loads(line)
        if r['prompt_idx'] in seen: raise RuntimeError('duplicate prompt_idx')
        seen.add(r['prompt_idx'])
        if len(r['samples']) != K: raise RuntimeError('sample count != 128')
        if not r['gold_tests'] or not all(isinstance(t, str) and t.strip() for t in r['gold_tests']):
            raise RuntimeError(f'empty tests prompt {r["prompt_idx"]}')
        r['_entry'] = r.get('entry_point') or EPS.get(r['prompt'])
        if r['source'] == 'humaneval' and not r['_entry']:
            raise RuntimeError('unresolved entry point')
        r['_tests'] = r['gold_tests'][0] if r['source'] == 'humaneval' else '\n'.join(r['gold_tests'])
        recs.append(r)
    return recs


def judge(tag):
    out = {}
    for line in open(f'{EVAL}/{tag}.slj_scores.jsonl', encoding='utf-8'):
        r = json.loads(line)
        if r['prompt_idx'] in out: raise RuntimeError('dup judge id')
        if len(r['scores']) != K or any((not math.isfinite(x)) or x < 0 or x > 1 for x in r['scores']):
            raise RuntimeError('bad judge scores')
        out[r['prompt_idx']] = [1 if x >= 1.0 else 0 for x in r['scores']]
    return out


def pct(v):
    v = sorted(v); lo = N_BOOT // 40
    return [v[lo], v[N_BOOT - lo - 1]]


def boot(e, j, strata, stratified):
    rng = random.Random(SEED); n = len(e); groups = {}
    for i, s in enumerate(strata): groups.setdefault(s, []).append(i)
    ge, gj, gg = [], [], []
    for _ in range(N_BOOT):
        if stratified:
            pick = []
            for m in groups.values(): pick += [m[rng.randrange(len(m))] for _ in m]
        else:
            pick = [rng.randrange(n) for _ in range(n)]
        a = sum(e[i] for i in pick) / n; b = sum(j[i] for i in pick) / n
        ge.append(a); gj.append(b); gg.append(b - a)
    return {'exec_ci95': pct(ge), 'judge_ci95': pct(gj), 'gap_ci95': pct(gg)}


def summarize(e, j, strata):
    n = len(e); per = {}
    for s in sorted(set(strata)):
        ix = [i for i, x in enumerate(strata) if x == s]
        es = [e[i] for i in ix]; js = [j[i] for i in ix]
        per[s] = {'n': len(ix), 'exec': sum(es) / len(ix), 'judge': sum(js) / len(ix),
                  'gap': (sum(js) - sum(es)) / len(ix), **boot(es, js, [s] * len(ix), False)}
    return {'n': n, 'exec': sum(e) / n, 'judge': sum(j) / n, 'gap': (sum(j) - sum(e)) / n,
            'pooled_bootstrap': boot(e, j, strata, False),
            'stratified_bootstrap': boot(e, j, strata, True), 'per_source': per}


def cluster_rate(counts, num_i, den_i):
    """prompt-cluster bootstrap CI for sum(num)/sum(den); counts = list of per-prompt tuples."""
    rng = random.Random(SEED); n = len(counts)
    pt = sum(c[num_i] for c in counts) / max(1, sum(c[den_i] for c in counts))
    vals = []
    for _ in range(N_BOOT):
        pick = [counts[rng.randrange(n)] for _ in range(n)]
        d = sum(c[den_i] for c in pick)
        vals.append(sum(c[num_i] for c in pick) / d if d else float('nan'))
    vals = sorted(v for v in vals if v == v); lo = len(vals) // 40
    if not vals:
        return {'rate': None, 'ci95': None, 'note': 'undefined: zero denominator in every resample'}
    return {'rate': pt, 'ci95': [vals[lo], vals[len(vals) - lo - 1]]}


def main():
    prov = {'python': platform.python_version(), 'protocol': {
        'timeout_s': 10.0,
        'humaneval': 'full problem prompt executed first, then extracted code, then tests and check(entry_point); token via os.write captured before candidate code',
        'mbpp': 'stdlib preamble + extracted code + asserts', 'mbpp_pre': MBPP_PRE,
        'adaptation': 'run-time alias-only; strict NameError on target; target absent; exactly one candidate function; no class; call shapes bind to real signature',
        'judge_pairing': 'ASSUMED sample j of judge cache = sample j of samples file; NOT independently verifiable from the cache (no sample hashes). Plausibility only.',
        'judge_input': 'one user message with the problem, the reference answer and the candidate response, in that order (rewards/llm_judge.py); cache-time messages were not retained',
        'security': 'not an adversarial boundary; do not use as sole evaluator for reward-optimized checkpoints'}}
    recs_all = {tag: load(tag) for tag, _ in RUNS}
    prov['sha256'] = {p: fhash(f'{REPO}/{p}') for p in ['rewards/code_execution.py', 'rewards/_extract.py', 'rewards/base.py', 'data/code/eval.json']}
    for tag, _ in RUNS:
        for suf in ('samples', 'slj_scores'):
            prov['sha256'][f'{tag}.{suf}.jsonl'] = fhash(f'{EVAL}/{tag}.{suf}.jsonl')

    base = load('base_qwen7b_code__code')
    controls = []
    for r in base:
        if r['source'] == 'humaneval':
            controls.append((('ref', r['prompt_idx']), '', r['prompt'] + (r.get('gold_answer') or ''), r['_tests'], r['_entry'], None, 10.0))
        else:
            controls.append((('ref', r['prompt_idx']), MBPP_PRE, r['gold_answer'], r['_tests'], None, None, 10.0))
    he = [r for r in base if r['source'] == 'humaneval'][:8]
    neg = []
    for r in he:
        ep = r['_entry']
        neg += [(('neg_none', r['prompt_idx']), pre_for(r), f'def {ep}(*a, **k):\n    return None\n', r['_tests'], ep, None, 10.0),
                (('neg_exit0', r['prompt_idx']), pre_for(r), 'import sys\nsys.exit(0)\n', r['_tests'], ep, None, 10.0),
                (('neg_osexit', r['prompt_idx']), pre_for(r), 'import os\nos._exit(0)\n', r['_tests'], ep, None, 10.0),
                (('neg_spoof', r['prompt_idx']), pre_for(r),
                 f'import json, sys, os\njson.dumps = lambda *a, **k: \'{{"ok": true}}\'\nsys.stdout.write = lambda *a, **k: None\ndef {ep}(*a, **k):\n    return None\n',
                 r['_tests'], ep, None, 10.0)]
    with mp.get_context('fork').Pool(16) as pool:
        cres = dict(pool.map(run_one, controls + neg, chunksize=8))
    srcmap = {r['prompt_idx']: r['source'] for r in base}
    ref_total = {'humaneval': [0, 0], 'mbpp': [0, 0]}; ref_fail = []
    for (kind, pidx), v in cres.items():
        if kind == 'ref':
            s = srcmap[pidx]; ref_total[s][1] += 1; ref_total[s][0] += int(v[0])
            if not v[0]: ref_fail.append((pidx, s, v[1], v[2]))
    neg_pass = {k: sum(int(v[0]) for (kk, p), v in cres.items() if kk == k) for k in ('neg_none', 'neg_exit0', 'neg_osexit', 'neg_spoof')}
    prov['controls'] = {'reference_pass(pass,total)': ref_total, 'reference_failures': ref_fail[:40],
                        'negative_control_passes_(expect_0)': neg_pass, 'n_neg_per_kind': len(he)}
    print('CONTROLS', json.dumps(prov['controls']), flush=True)
    if ref_fail or any(neg_pass.values()):
        raise RuntimeError('control failure -> aborting (see CONTROLS line)')

    results = []
    for tag, label in RUNS:
        recs = recs_all[tag]; jd = judge(tag)
        if set(jd) != {r['prompt_idx'] for r in recs}: raise RuntimeError('id mismatch judge/samples')
        by = {r['prompt_idx']: r for r in recs}
        jobs = []
        for r in recs:
            pre = pre_for(r)
            for j, s in enumerate(r['samples']):
                code = norm_code(s)
                if code:
                    jobs.append(((r['prompt_idx'], j), pre, code, r['_tests'], r['_entry'] if r['source'] == 'humaneval' else None, None, 10.0))
        print(tag, 'strict jobs', len(jobs), flush=True)
        with mp.get_context('fork').Pool(16) as pool:
            strict = dict(pool.map(run_one, jobs, chunksize=32))
        ajobs, tgt_of = [], {}
        for (p, j), pre, code, tests, entry, _, to in jobs:
            r = by[p]
            if r['source'] != 'mbpp' or strict[(p, j)][0]:
                continue
            if p not in tgt_of:
                tgt_of[p] = target_and_shapes(r['gold_tests'])
            tgt, shapes = tgt_of[p]
            if tgt and strict[(p, j)][1] == 'NameError' and strict[(p, j)][2] == tgt:
                ajobs.append(((p, j), pre, code, tests, entry, (tgt, shapes), to))
        print(tag, 'adapted jobs', len(ajobs), flush=True)
        with mp.get_context('fork').Pool(16) as pool:
            adapted = dict(pool.map(run_one, ajobs, chunksize=32)) if ajobs else {}
        ok_s = {k: v[0] for k, v in strict.items()}
        ok_a = dict(ok_s)
        for k, v in adapted.items(): ok_a[k] = v[0]
        pidx = sorted(by); strata = [by[p]['source'] for p in pidx]
        e_s = [1 if any(ok_s.get((p, j), False) for j in range(K)) else 0 for p in pidx]
        e_a = [1 if any(ok_a.get((p, j), False) for j in range(K)) else 0 for p in pidx]
        jp = [1 if any(jd[p]) else 0 for p in pidx]
        cls = {'humaneval': {}, 'mbpp': {}}; syms = {'humaneval': {}, 'mbpp': {}}
        for (p, j), (ok, err, sym) in strict.items():
            if ok: continue
            s = by[p]['source']
            tgt = target_and_shapes(by[p]['gold_tests'])[0] if s == 'mbpp' else by[p]['_entry']
            key = err + ((':target' if sym == tgt else ':other') if err == 'NameError' else '')
            cls[s][key] = cls[s].get(key, 0) + 1
            if key == 'NameError:other':
                syms[s][sym] = syms[s].get(sym, 0) + 1
        top_syms = {s: sorted(d.items(), key=lambda kv: -kv[1])[:15] for s, d in syms.items()}
        a_blocked = sum(1 for v in adapted.values() if v[1] == '_AliasBlocked')
        nocode = sum(1 for r in recs for s in r['samples'] if not norm_code(s))
        conf = {}
        rates = {}
        for mode, ok in (('strict', ok_s), ('adapted', ok_a)):
            conf[mode] = {'humaneval': [[0, 0], [0, 0]], 'mbpp': [[0, 0], [0, 0]]}
            per = {'humaneval': [], 'mbpp': []}
            for p in pidx:
                s = by[p]['source']; c = [0, 0, 0, 0]  # exec0, exec0&judge1, exec1, exec1&judge0
                for j in range(K):
                    e = int(ok.get((p, j), False)); jj = jd[p][j]
                    conf[mode][s][e][jj] += 1
                    if e == 0: c[0] += 1; c[1] += jj
                    else: c[2] += 1; c[3] += (1 - jj)
                per[s].append(tuple(c))
            rates[mode] = {s: {'P(judge_correct | exec_fail)': cluster_rate(per[s], 1, 0),
                               'P(judge_wrong | exec_pass)': cluster_rate(per[s], 3, 2)} for s in per}
        newly = [p for p, a, s in zip(pidx, e_a, e_s) if a and not s]
        entry = {'run_tag': tag, 'label': label, 'strict': summarize(e_s, jp, strata), 'adapted': summarize(e_a, jp, strata),
                 'adaptation': {'eligible_strict_nameerror_target_samples': len(ajobs),
                                'blocked_at_runtime': a_blocked,
                                'adapted_samples_that_pass': sum(1 for v in adapted.values() if v[0]),
                                'prompts_newly_passing_mbpp': len(newly), 'newly_passing_prompt_idxs': newly},
                 'error_classes_strict_failing_samples': cls, 'top_NameError_other_symbols': top_syms,
                 'samples_without_extractable_code': nocode, 'tag_line_samples_normalized': sum(1 for r in recs for x in r['samples'] if (_extract.extract_code(x or '').partition(chr(10))[0].strip().lower() in ('python','py','python3'))),
                 'sample_level_confusion_[exec][judge]': conf, 'sample_level_rates_cluster_bootstrap': rates}
        results.append(entry)
        print('RUN', tag, json.dumps({'adaptation': entry['adaptation'] | {'newly_passing_prompt_idxs': len(newly)}}), flush=True)
        json.dump({'prompt_idxs': pidx, 'strict_ok': [[int(ok_s.get((p, j), False)) for j in range(K)] for p in pidx],
                   'adapted_ok': [[int(ok_a.get((p, j), False)) for j in range(K)] for p in pidx],
                   'strict_err': [[strict.get((p, j), (False, 'NoCode', ''))[1] for j in range(K)] for p in pidx],
                   'strict_sym': [[strict.get((p, j), (False, '', ''))[2] for j in range(K)] for p in pidx],
                   'source': strata, 'judge_ok': [jd[p] for p in pidx]},
                  open(f'{HARD}/{tag}.canon5_sample_level.json', 'w'))
        prov['sha256'][f'{tag}.canon5_sample_level.json'] = fhash(f'{HARD}/{tag}.canon5_sample_level.json')
    prov['sha256']['scripts/canon5.py'] = fhash(f'{REPO}/scripts/canon5.py')
    json.dump({'provenance': prov, 'results': results}, open(OUT, 'w'), indent=2)
    print('WROTE', OUT, flush=True)


if __name__ == '__main__':
    main()
