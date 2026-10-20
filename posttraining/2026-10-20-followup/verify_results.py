"""Verify a source-only snapshot and recount stored comparator verdicts on CPU."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT/'results/2026-10-20-qwen-validation'
sys.path.insert(0, str(HERE/'qwen'))
from compare_final_evaluation import EXPECTED, paired_bootstrap, wilson


def require(value, message):
    if not value:
        raise ValueError(message)


def close(actual, expected):
    require(math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-11),
            'Numeric result differs from verdict recount')


def load(path):
    return json.loads(path.read_text())


def verify():
    manifest = load(RESULTS/'publication-manifest.json')
    for name, item in manifest['files'].items():
        path = ROOT/name
        require(not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(ROOT),
                'Missing or escaping publication path: '+name)
        raw = path.read_bytes()
        require(len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'],
                'Publication hash mismatch: '+name)
    provenance = load(HERE/'source-provenance.json')
    adaptations = 0
    for name, item in provenance['files'].items():
        raw = (HERE/name).read_bytes()
        identity = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        require(identity == item['published'], 'Source publication hash mismatch: '+name)
        if not item['publication_adaptations']:
            require(identity == item['original'], 'Unrecorded source adaptation: '+name)
        else:
            adaptations += 1
    scores = load(RESULTS/'metrics.json')
    require(scores['status'] == 'verified_complete', 'Incomplete comparator result')
    frozen = ROOT/'posttraining/2026-10-20-video-model/scripts'
    for name, expected in scores['frozen_sources'].items():
        require(hashlib.sha256((frozen/name).read_bytes()).hexdigest() == expected,
                'Frozen evaluator/generator source differs: '+name)
    protocol = scores['protocol']
    require((protocol['evaluation_protocol'], protocol['seed'], protocol['context'],
             protocol['prompt_token_cap'], protocol['max_new_tokens'], protocol['generation']) ==
            ('v2-hashlib-entrypoint', 20261019, 2048, 1100, 1800, 'greedy'),
            'Unexpected comparison protocol')
    with (RESULTS/'paired-verdicts.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == sum(EXPECTED.values()), 'Wrong total question count')
    seen, prompts = set(), set()
    labels = ['base','posttrained','qwen-0.5b','qwen-1.5b']
    for row in rows:
        key = row['task'], row['id']
        prompt = row['task'], row['prompt_sha256']
        require(key not in seen and prompt not in prompts, 'Duplicate task ID or prompt')
        for field in ['prompt_sha256','content_sha256']:
            require(re.fullmatch('[a-f0-9]{64}', row[field]) is not None, 'Invalid content hash')
        for label in labels:
            require(row[label+'_correct'] in ['0','1'], 'Non-boolean verdict')
        seen.add(key)
        prompts.add(prompt)
    for task, n in EXPECTED.items():
        items = [row for row in rows if row['task'] == task]
        require(len(items) == n, 'Wrong task count: '+task)
        for label in labels:
            correct = sum(int(row[label+'_correct']) for row in items)
            score = scores['metrics'][label][task]
            require((correct,n) == (score['correct'],score['n']), 'Stored score differs')
            close(score['accuracy'], correct/n)
            for actual, expected in zip(score['wilson_95_ci'], wilson(correct,n)):
                close(actual,expected)
    for j, label in enumerate(['qwen-0.5b','qwen-1.5b']):
        for i, (task,n) in enumerate(EXPECTED.items()):
            items = [row for row in rows if row['task'] == task]
            gains = sum(row['posttrained_correct']=='0' and row[label+'_correct']=='1' for row in items)
            losses = sum(row['posttrained_correct']=='1' and row[label+'_correct']=='0' for row in items)
            delta = scores['qwen_minus_posttrained'][label][task]
            require((gains,losses) == (delta['qwen_gained_items'],delta['qwen_lost_items']),
                    'Paired gains/losses differ')
            close(delta['qwen_minus_posttrained_pp'],100*(gains-losses)/n)
            for actual, expected in zip(delta['paired_bootstrap_95_ci_pp'],
                                        paired_bootstrap(gains,losses,n,20261029+10*j+i)):
                close(actual,expected)
    prep = load(RESULTS/'preparation-outcome.json')
    for field in ['new_48h_budget_started','GPU_preflight_executed','GPU_training_executed','three_pilot_results_exist']:
        require(prep[field] is False, 'Unlaunched pilot represented as a completed run')
    require(prep['CPT']['production_native_objectives_created'] is False and
            prep['CPT']['quality_approved'] is False, 'Raw candidate counts represented as approved data')
    return dict(status='verified', comparator_models=4, paired_questions=len(rows),
                source_files=len(provenance['files']), publication_adaptations=adaptations,
                scores_and_intervals_recomputed=True, new_inference_or_grading=False,
                raw_output_grading_independently_verified_by_this_command=False,
                validation_pilots_launched=False, Chat_training_launched=False)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
