"""Judge-pairing re-check. Run while a vLLM server with Qwen2.5-72B-Instruct is reachable at URL (below).

Seeded subset: 80 prompts per source (HumanEval, MBPP) per run x 8 random samples (prioritizes prompt count).
Messages rebuilt with the repo's LLMJudgeReward (same template; gold dict built as in the original scoring), temperature 0, max_tokens 4.
Recorded per call: sample sha256, message sha256, chat-template token count, RAW text, production-parsed verdict,
parse validity, ALL attempt errors, chosen comparison prompt. No exception can abort the run or be silently dropped.
Compared against cached slj score at the same (prompt, sample), shifted-by-one sample, and a pre-chosen other prompt.
Prompt-cluster bootstrap CIs for agreement improvements over baselines. Context-limit failures reported separately.
"""
import hashlib, json, os, random, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor

REPO = os.environ.get('EVOROBUST_REPO', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL = os.environ.get('EVOROBUST_SAMPLES', f'{REPO}/data/samples')
sys.path.insert(0, REPO)
from rewards.llm_judge import LLMJudgeReward, _parse_binary, JUDGE_PROMPT_TEMPLATE

URL = 'http://localhost:8000/v1'
MODEL = 'Qwen/Qwen2.5-72B-Instruct'
SERVER_MAX_LEN = 4096
RUNS = ['base_qwen7b_code__code', 'base_llama8b__code']
N_PER_SOURCE = 80
K_SAMPLES = 8
SEED = 7
OUTF = f'{REPO}/tables/judge_recheck_summary.json'
ROWF = f'{EVAL}/judge_recheck.rows.json'
MSGF = f'{EVAL}/judge_recheck.messages.jsonl'


def h(x):
    return hashlib.sha256(x.encode('utf-8')).hexdigest()


def pct(v):
    v = sorted(v); lo = len(v) // 40
    return [v[lo], v[len(v) - lo - 1]]


def main():
    judge = LLMJudgeReward(judge_model=MODEL, domain='code', vllm_endpoint=URL)
    tok = judge._tokenizer
    if tok is None:
        raise SystemExit('the Qwen2.5 tokenizer is required to measure request lengths')
    rng = random.Random(SEED)
    items = []
    for tag in RUNS:
        recs, cache = {}, {}
        for line in open(f'{EVAL}/{tag}.samples.jsonl', encoding='utf-8'):
            r = json.loads(line)
            if r['prompt_idx'] in recs: raise RuntimeError('duplicate sample id')
            if len(r['samples']) != 128: raise RuntimeError('sample count')
            recs[r['prompt_idx']] = r
        for line in open(f'{EVAL}/{tag}.slj_scores.jsonl', encoding='utf-8'):
            r = json.loads(line)
            if r['prompt_idx'] in cache: raise RuntimeError('duplicate judge id')
            if len(r['scores']) != 128 or any(x not in (0, 1, 0.0, 1.0) for x in r['scores']): raise RuntimeError('judge scores')
            cache[r['prompt_idx']] = r['scores']
        if set(recs) != set(cache): raise RuntimeError('id sets differ')
        for src in ('humaneval', 'mbpp'):
            pids = sorted(p for p, r in recs.items() if r['source'] == src)
            for p in rng.sample(pids, N_PER_SOURCE):
                other = rng.choice([q for q in pids if q != p])
                for j in rng.sample(range(128), K_SAMPLES):
                    items.append(dict(tag=tag, p=p, j=j, src=src, rec=recs[p], cached=cache[p], other=other,
                                      other_scores=cache[other]))
    print('items', len(items), flush=True)

    def call(it):
        row = dict(tag=it['tag'], prompt_idx=it['p'], sample_idx=it['j'], source=it['src'],
                   other_prompt=it['other'], raw=None, parsed=None, parse_valid=None, attempt_errors=[], fatal=None,
                   n_input_tokens=None, cached_same=int(it['cached'][it['j']] >= 1.0),
                   cached_shift1=int(it['cached'][(it['j'] + 1) % 128] >= 1.0),
                   cached_other_prompt=int(it['other_scores'][it['j']] >= 1.0))
        msgs = None
        try:
            r = it['rec']
            gold = {'answer': r.get('gold_answer'), 'tests': r.get('gold_tests'), 'source': r.get('source'),
                    'entry_point': r.get('entry_point')}
            sample = r['samples'][it['j']]
            row['sample_sha256'] = h(sample)
            msgs = judge._build_messages(r['prompt'], sample, gold)
            row['message_sha256'] = h(json.dumps(msgs, sort_keys=True))
            if tok is not None:
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
                row['n_input_tokens'] = len(ids)
            for attempt in range(3):
                try:
                    row['raw'] = judge._post(msgs); break
                except Exception as e:  # noqa: BLE001
                    row['attempt_errors'].append(f'{type(e).__name__}: {str(e)[:200]}'); time.sleep(2)
            if row['raw'] is not None:
                row['parsed'] = int(_parse_binary(row['raw']))
                row['parse_valid'] = row['raw'].strip() in ('0', '1')
        except Exception:  # noqa: BLE001
            row['fatal'] = traceback.format_exc()[-400:]
        return row, msgs

    with ThreadPoolExecutor(16) as ex:
        out = list(ex.map(call, items))
    rows = [o[0] for o in out]
    with open(MSGF, 'w', encoding='utf-8') as f:
        for row, msgs in out:
            f.write(json.dumps({'tag': row['tag'], 'prompt_idx': row['prompt_idx'], 'sample_idx': row['sample_idx'], 'messages': msgs}) + '\n')

    failed = [r for r in rows if r['raw'] is None]
    over = [r for r in rows if r['n_input_tokens'] is None or r['n_input_tokens'] + 4 > SERVER_MAX_LEN]
    valid = [r for r in rows if r['raw'] is not None and r['parse_valid'] and r['n_input_tokens'] is not None
             and r['n_input_tokens'] + 4 <= SERVER_MAX_LEN]

    def rate(rs, key):
        return sum(1 for r in rs if r['parsed'] == r[key]) / len(rs) if rs else None

    def cluster_diff(rs, k1, k2):
        by = {}
        for r in rs: by.setdefault((r['tag'], r['prompt_idx']), []).append(int(r['parsed'] == r[k1]) - int(r['parsed'] == r[k2]))
        cl = list(by.values()); rb = random.Random(1); vals = []
        for _ in range(5000):
            pick = [cl[rb.randrange(len(cl))] for _ in cl]
            n = sum(len(c) for c in pick)
            vals.append(sum(sum(c) for c in pick) / n)
        return {'diff': sum(sum(c) for c in cl) / sum(len(c) for c in cl), 'ci95': pct(vals)}

    groups = {}
    for r in valid: groups.setdefault((r['tag'], r['source']), []).append(r)
    summ = {
        'n_items': len(rows), 'n_failed_requests': len(failed), 'fatal_rows': sum(1 for r in rows if r['fatal']),
        'failure_examples': [(r['attempt_errors'] or [r['fatal']])[0] for r in failed[:5]],
        'n_over_server_context_(chat-template tokens + 4 > 4096)': len(over),
        'n_invalid_parse': sum(1 for r in rows if r['raw'] is not None and not r['parse_valid']),
        'n_valid_within_context': len(valid),
        'max_input_tokens': max((r['n_input_tokens'] or 0) for r in rows),
        'agreement_same_sample': rate(valid, 'cached_same'), 'agreement_shift1': rate(valid, 'cached_shift1'),
        'agreement_other_prompt': rate(valid, 'cached_other_prompt'),
        'improvement_same_minus_shift1': cluster_diff(valid, 'cached_same', 'cached_shift1') if valid else None,
        'improvement_same_minus_other_prompt': cluster_diff(valid, 'cached_same', 'cached_other_prompt') if valid else None,
        'by_run_source': {f'{k[0]}|{k[1]}': {'n': len(v), 'same': rate(v, 'cached_same'), 'shift1': rate(v, 'cached_shift1'),
                                              'other_prompt': rate(v, 'cached_other_prompt')} for k, v in groups.items()},
        'cached_zero_where_new_one_(n)': sum(1 for r in valid if r['cached_same'] == 0 and r['parsed'] == 1),
        'cached_one_where_new_zero_(n)': sum(1 for r in valid if r['cached_same'] == 1 and r['parsed'] == 0),
        'over_context_items_cached_verdict_counts': {'cached0': sum(1 for r in over if r['cached_same'] == 0), 'cached1': sum(1 for r in over if r['cached_same'] == 1)},
        'template_sha256': h(JUDGE_PROMPT_TEMPLATE), 'model': MODEL, 'seed': SEED,
        'design': f'{N_PER_SOURCE} prompts per source x {K_SAMPLES} random samples, per run',
        'note': 'Cannot prove identity with the original cache-time messages; evidence of consistency only.',
    }
    json.dump(summ, open(OUTF, 'w'), indent=1)
    json.dump(rows, open(ROWF, 'w'))
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == '__main__':
    main()
