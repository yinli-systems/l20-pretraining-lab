"""CPU-only audit of existing RFT data and development outputs; no generation."""
import collections
import concurrent.futures
import datetime
import hashlib
import json
import math
import multiprocessing
import random
from pathlib import Path

import numpy as np

from common import CACHE, ROOT, SEED, encode_messages, tokenizer, write_json
from evaluate import code_correct, extract_code, math_correct


def math_job(item):
    text, answer = item
    return math_correct(text, answer)


def tests_text(row):
    tests = row['tests']
    if isinstance(tests, str):
        tests = json.loads(tests)
    return '\n'.join(tests)


def code_control(item):
    row, kind = item
    text = '' if kind == 'empty' else extract_code(row['completion' if kind == 'generated' else 'reference'])
    return {'prompt_hash': row['prompt_hash'], 'kind': kind,
            'passed': code_correct(text, tests_text(row), '')}


def grade_dev(row):
    return code_correct(extract_code(row['completion'], row['prompt'], False, row['entry_point']),
                        row['test'], row['entry_point'])


def main():
    started = datetime.datetime.now(datetime.timezone.utc)
    report = {'started_utc': started.isoformat(), 'cpu_only': True,
              'new_generation': False, 'training_or_scoring_changes': False}
    accepted_path = ROOT / 'rft-1-accepted.jsonl'
    accepted = [json.loads(x) for x in accepted_path.read_text().splitlines()]
    report['source_counts'] = dict(collections.Counter(x['source'] for x in accepted))
    report['accepted_prompts'] = len(accepted)
    report['unique_prompt_hashes'] = len({x['prompt_hash'] for x in accepted})
    report['source_sha256'] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ['rft.py', 'train.py', 'common.py', 'evaluate.py', 'sandbox.py',
                     'rft-1-accepted.jsonl', 'rft-1-rollouts.jsonl']}

    tok = tokenizer()
    rng = random.Random(SEED + 1)
    pack = {}
    split_hashes = {}
    for split in ['train', 'valid']:
        chosen = [r for i, r in enumerate(accepted) if (i % 20 == 0) == (split == 'valid')]
        split_hashes[split] = {x['prompt_hash'] for x in chosen}
        sequences = []
        stats = collections.defaultdict(lambda: {'examples': 0, 'supervised_tokens': 0})
        skipped = collections.Counter()
        lengths = []
        for row in chosen:
            lengths.append(len(tok.encode(row['completion'], add_special_tokens=False)))
            for kind in ['completion', 'reference']:
                ids, labels = encode_messages(tok, [{'role': 'user', 'content': row['prompt']},
                                                     {'role': 'assistant', 'content': row[kind]}])
                if not 32 <= len(ids) <= 2048:
                    skipped[row['source'] + ':' + kind] += 1
                    continue
                mask = [int(y != -100) for y in labels]
                key = row['source'] + ':' + kind
                stats[key]['examples'] += 1
                stats[key]['supervised_tokens'] += sum(mask)
                sequences.append((ids, mask))
        rng.shuffle(sequences)
        blocks = []
        current = ([], [])
        for ids, mask in sequences:
            if len(current[0]) + len(ids) > 2048:
                blocks.append(current)
                current = ([], [])
            current[0].extend(ids)
            current[1].extend(mask)
        if current[0]:
            blocks.append(current)
        expected_ids = [np.asarray(ids + [tok.pad_token_id] * (2048-len(ids)), dtype=np.uint16)
                        for ids, _ in blocks]
        expected_mask = [np.asarray(mask + [0] * (2048-len(mask)), dtype=np.uint8)
                         for _, mask in blocks]
        original_ids = np.memmap(CACHE/'packed'/f'{split}.ids.u16', mode='r', dtype=np.uint16).reshape(-1, 2048)
        original_mask = np.memmap(CACHE/'packed'/f'{split}.mask.u8', mode='r', dtype=np.uint8).reshape(-1, 2048)
        indices = rng.sample(range(len(original_ids)), min(len(original_ids), max(4, int(len(blocks)*0.3))))
        expected_ids.extend(original_ids[i] for i in indices)
        expected_mask.extend(original_mask[i] for i in indices)
        actual_ids = np.memmap(CACHE/'rft-1'/f'{split}.ids.u16', mode='r', dtype=np.uint16).reshape(-1, 2048)
        actual_mask = np.memmap(CACHE/'rft-1'/f'{split}.mask.u8', mode='r', dtype=np.uint8).reshape(-1, 2048)
        assert len(actual_ids) == len(expected_ids)
        id_mismatch = [i for i, x in enumerate(expected_ids) if not np.array_equal(x, actual_ids[i])]
        mask_mismatch = [i for i, x in enumerate(expected_mask) if not np.array_equal(x, actual_mask[i])]
        assert not id_mismatch and not mask_mismatch
        assert np.all(actual_ids < len(tok))
        pack[split] = {'accepted_prompts': len(chosen), 'sequences_by_source_and_kind': dict(stats),
                       'skipped_by_source_and_kind': dict(skipped), 'rft_blocks': len(blocks),
                       'replay_blocks': len(indices), 'total_blocks': len(actual_ids),
                       'replay_block_fraction': len(indices)/len(actual_ids),
                       'replay_supervised_tokens': int(actual_mask[len(blocks):].sum()),
                       'all_supervised_tokens': int(actual_mask.sum()),
                       'id_mismatches': id_mismatch, 'mask_mismatches': mask_mismatch,
                       'completion_token_lengths': {'min': min(lengths), 'max': max(lengths),
                           'median': sorted(lengths)[len(lengths)//2],
                           'at_least_1023_tokens': sum(n >= 1023 for n in lengths)},
                       'packed_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [CACHE/'rft-1'/f'{split}.ids.u16', CACHE/'rft-1'/f'{split}.mask.u8']}}
    report['packing'] = pack
    report['rft_train_valid_prompt_overlap'] = len(split_hashes['train'] & split_hashes['valid'])

    dev = {}
    for name in ['sft-12000', 'rft-1-23']:
        rows = [json.loads(x) for x in (ROOT/'evaluations'/f'{name}.samples.jsonl').read_text().splitlines()]
        math_rows = [x for x in rows if x['task'] != 'humaneval']
        code_rows = [x for x in rows if x['task'] == 'humaneval']
        with concurrent.futures.ProcessPoolExecutor(max_workers=2,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            math_ok = list(pool.map(math_job, [(x['completion'], x['answer']) for x in math_rows]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            code_ok = list(pool.map(grade_dev, code_rows))
        checks = list(zip(math_rows, math_ok)) + list(zip(code_rows, code_ok))
        dev[name] = {'regraded_rows': len(checks),
                     'disagreements': [{'task': x['task'], 'id': x['id'], 'old': x['correct'], 'new': ok}
                                       for x, ok in checks if x['correct'] != ok]}
    report['development_regrading'] = dev

    audit_rng = random.Random(20261020)
    controls = {}
    examples = {}
    for source in ['math', 'code']:
        rows = [r for r in accepted if r['source'] == source]
        sample = audit_rng.sample(rows, min(64, len(rows)))
        examples[source] = [{k: x[k] for k in ['prompt_hash', 'prompt', 'answer', 'completion', 'reference', 'tests']}
                            for x in random.Random(20261020).sample(rows, 5)]
        if source == 'math':
            with concurrent.futures.ProcessPoolExecutor(max_workers=2,
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                generated = list(pool.map(math_job, [(x['completion'], x['answer']) for x in sample]))
                reference = list(pool.map(math_job, [(x['reference'], x['answer']) for x in sample]))
            controls[source] = {'n': len(sample), 'generated_answer_matches': sum(generated),
                                 'reference_answer_matches': sum(reference),
                                 'scope': 'Answer equivalence only; not proof validity or full reasoning verification.'}
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(code_control, [(x, k) for x in sample
                                    for k in ['generated', 'reference', 'empty']]))
            controls[source] = {'n': len(sample),
                'passed_by_kind': {k: sum(x['passed'] for x in results if x['kind'] == k)
                                  for k in ['generated', 'reference', 'empty']}, 'items': results}
    report['training_controls'] = controls
    report['fixed_random_sample_examples'] = examples

    import sympy as sp
    x = sp.Symbol('x', real=True)
    roots = sp.solve(2*x*x-15*x-2, x)
    report['confirmed_bad_math_training_examples'] = [
        {'prompt_hash': '323418120735577640edf98328e119d9ef3b6fa572e09f365bce1559f90870e3',
         'issue': 'Inconsistent premises: additivity and f(1)=2 require f(2)=4, contradicting f(2)=5.',
         'given_answer_accepted_by_outcome_grader': '7'},
        {'prompt_hash': '14fd765cf7c4d33eb5e74113569f26cead79db33e80ca8cc134115c14abbd8ce',
         'issue': 'Both quadratic roots satisfy the original equation; generated answer incorrectly discards the negative root.',
         'verified_roots': [str(t) for t in roots],
         'both_satisfy_equation': all(sp.simplify(2*t*t-15*t+9-11) == 0 for t in roots)}]
    assert all(any(x['prompt_hash'] == e['prompt_hash'] for x in accepted)
               for e in report['confirmed_bad_math_training_examples'])
    training = [json.loads(x) for x in (ROOT/'logs/rft-1-metrics.jsonl').read_text().splitlines()]
    report['logged_training_health'] = {'logged_steps': [x['step'] for x in training],
        'all_logged_losses_and_gradient_norms_finite': all(math.isfinite(x['loss']) and math.isfinite(x['gradient_norm']) for x in training),
        'gradient_norms': [x['gradient_norm'] for x in training],
        'completed': json.loads((ROOT/'rft-1-complete.json').read_text())}
    report['limitations'] = ['No causal ablation was run; data-quality defects do not establish their quantitative contribution to the score drop.',
        'The accepted-data controls sample 64 per source and do not certify every solution or every requirement.',
        'Completion length is measured by retokenizing decoded text; finish reasons were not recorded, so truncation cannot be certified retrospectively.',
        'No final test partition was accessed. No GPU generation or training restart was performed.']
    report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(ROOT/'audit-rft-regression.json', report)
    print(json.dumps({'finished_utc': report['finished_utc'], 'source_counts': report['source_counts'],
        'packing': pack, 'regrading': dev, 'training_controls': {k: {a: b for a,b in v.items() if a != 'items'} for k,v in controls.items()}}), flush=True)


if __name__ == '__main__':
    main()
