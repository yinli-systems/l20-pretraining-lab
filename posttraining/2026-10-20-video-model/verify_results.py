"""Replay published post-training verdicts and intervals; never run inference."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT/'results/2026-10-20-math-code'
sys.path.insert(0, str(HERE/'scripts'))
from compare_final_evaluation import EXPECTED, PROTOCOL, paired_bootstrap, wilson


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    require(math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-11),
            f'Numeric mismatch: {actual} != {expected}')


def load(path):
    return json.loads(path.read_text())


def verify():
    manifest = load(RESULTS/'source-manifest.json')
    for relative, record in manifest['files'].items():
        p = ROOT/relative
        require(not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(ROOT),
                'Missing or escaping file: '+relative)
        raw = p.read_bytes()
        require(len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256'],
                'File hash/size mismatch: '+relative)
    provenance = load(HERE/'source-provenance.json')
    for name, record in provenance['files'].items():
        require(hashlib.sha256((HERE/'scripts'/name).read_bytes()).hexdigest() == record['sha256'],
                'Run source mismatch: '+name)
    original = load(RESULTS/'evidence/original-input-hashes.json')['files']
    for name in ['evaluate.py','sandbox.py','common.py','pipeline.py']:
        require(provenance['files'][name]['sha256'] == original[name]['sha256'],
                'Frozen runtime source differs: '+name)
    m = load(RESULTS/'metrics.json')
    require(m['status'] == 'verified_complete', 'Incomplete final result')
    require(m['selected_checkpoint']['stage'] == 'sft' and m['selected_checkpoint']['step'] == 12000,
            'Wrong selected checkpoint')
    require(m['protocol'] == PROTOCOL, 'Protocol changed')
    for label in ['base-final', 'best-final']:
        meta = load(RESULTS/'evidence'/(label+'.json'))
        for key, value in PROTOCOL.items():
            require(meta[key] == value, 'Evaluation protocol mismatch: '+key)
        require(meta['base_prompting'] == (label == 'base-final'), 'Wrong prompt interface')
    with (RESULTS/'paired-verdicts.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == sum(EXPECTED.values()), 'Incomplete paired verdict file')
    seen, prompts = set(), set()
    for row in rows:
        key = (row['task'], row['id'])
        prompt = (row['task'], row['prompt_sha256'])
        require(key not in seen and prompt not in prompts, 'Duplicate task ID or prompt')
        require(bool(re.fullmatch('[a-f0-9]{64}', row['prompt_sha256'])), 'Invalid prompt hash')
        require(row['base_correct'] in ['0','1'] and row['posttrained_correct'] in ['0','1'],
                'Invalid verdict')
        seen.add(key); prompts.add(prompt)
    for i, (task, n) in enumerate(EXPECTED.items()):
        items = [r for r in rows if r['task'] == task]
        require(len(items) == n, 'Wrong task count: '+task)
        for label, field in [('base-final','base_correct'), ('best-final','posttrained_correct')]:
            k = sum(int(r[field]) for r in items)
            score = m['metrics'][label][task]
            require((score['correct'], score['n']) == (k,n), 'Stored count mismatch')
            close(score['accuracy'], k/n)
            for actual, wanted in zip(score['wilson_95_ci'], wilson(k,n)):
                close(actual,wanted)
            meta = load(RESULTS/'evidence'/(label+'.json'))['metrics'][task]
            require((meta['correct'],meta['n']) == (k,n), 'Original summary count mismatch')
            close(meta['accuracy'], k/n)
        gains = sum(r['base_correct']=='0' and r['posttrained_correct']=='1' for r in items)
        losses = sum(r['base_correct']=='1' and r['posttrained_correct']=='0' for r in items)
        delta = m['deltas'][task]
        require((gains,losses) == (delta['gained_items'],delta['lost_items']), 'Paired counts differ')
        close(delta['delta_percentage_points'],100*(gains-losses)/n)
        for actual,wanted in zip(delta['paired_bootstrap_95_ci_pp'],paired_bootstrap(gains,losses,n,20261019+i)):
            close(actual,wanted)
    for label in ['base-final','best-final']:
        close(m['macro_accuracy'][label],sum(s['accuracy'] for s in m['metrics'][label].values())/3)
    return {'status':'verified','paired_questions':len(rows),
            'source_files':len(provenance['files']), 'bound_files':len(manifest['files']),
            'scores_and_intervals_recomputed':True, 'new_inference_or_grading':False,
            'raw_output_grading_independently_verified_by_this_command':False}


if __name__ == '__main__':
    print(json.dumps(verify(),indent=2))
