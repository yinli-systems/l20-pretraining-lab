"""Distinct source repair: complete computational calculus finals with CAS gold.

The failed broad R1 candidate is never a training input. This separate subset
requires one exactly parsed univariate derivative/integral task and independent
symbolic verification from that task, not agreement with its supplied label.
It remains quarantined pending a new fixed200 premise/trace review. Restricting
the math distribution is an explicit pilot-data limitation, not Qwen parity.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import collections
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import signal
import sys
import time

JOB = Path('/workspace/video-model/validation-48h')
WORK = Path('/dev/shm/video-model-validation48')
sys.path.insert(0, str(JOB / 'vendor'))


def expired(signum, frame):
    raise TimeoutError('Bounded CAS computation')


def expressions(prompt):
    pattern = r'(?s)\$\$(.*?)\$\$|\\\[(.*?)\\\]|\$(.*?)\$|\\\((.*?)\\\)'
    matches = list(re.finditer(pattern, prompt))
    # Additional formulae/conditions cannot be silently omitted from the task.
    if len(matches) != 1:
        return None
    return next(x for x in matches[0].groups() if x is not None).strip()


def certify(row):
    import sympy as sp
    from sympy.parsing.latex import parse_latex
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(8)
    try:
        prompt, final = row['prompt'], row['response']
        if re.search(r'(?i)prove|show that|maximum|minimum|smallest|largest|integer|parameter|partial|second derivative|nth|likely|intended|typo', prompt + ' ' + final):
            return None
        latex = expressions(prompt)
        if latex is None or len(latex) > 400:
            return None
        # Unknown functions, branches and multi-variable conditions are outside
        # this deliberately narrow repair, even when the source label matches.
        task = parse_latex(latex)
        x = sp.Symbol('x')
        if task.free_symbols - {x}:
            # y=f(x) is common for derivative questions; only its RHS is used.
            if isinstance(task, sp.Equality) and task.lhs == sp.Symbol('y') and not task.rhs.free_symbols - {x}:
                task = task.rhs
            else:
                return None
        answer = parse_latex(row['answer'])
        if isinstance(answer, sp.Equality):
            return None
        if answer.free_symbols - {x, sp.Symbol('C'), sp.Symbol('c')}:
            return None
        if isinstance(task, sp.Integral):
            if len(task.limits) != 1 or task.variables != [x] or task.function.has(sp.Integral):
                return None
            if len(task.limits[0]) == 1:
                residual = sp.simplify(sp.diff(answer, x) - task.function)
                mode = 'indefinite-integral-by-exact-differentiation'
            else:
                exact = task.doit()
                if exact.has(sp.Integral) or exact.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
                    return None
                residual = sp.simplify(exact - answer)
                mode = 'definite-integral-independent-exact-computation'
        elif re.search(r'(?i)derivative|differentiate', prompt):
            if task.has(sp.Integral) or isinstance(task, sp.Equality):
                return None
            residual = sp.simplify(sp.diff(task, x) - answer)
            mode = 'derivative-independent-symbolic-differentiation'
        else:
            return None
        if residual != 0:
            return None
        value = dict(row)
        value.update(source_repair='computational-calculus-cas-v1',
                     cas_mode=mode, independently_parsed_task=sp.srepr(task),
                     independent_answer_residual=sp.srepr(residual),
                     sympy_version=sp.__version__,
                     independent_computational_gold_verified=True,
                     all_reasoning_steps_certified=False,
                     split_assignment='quarantined_pending_fixed200_semantic_review_and_decontamination')
        return value
    except Exception:
        return None
    finally:
        signal.alarm(0)


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    parent = json.loads((JOB / 'openr1-short-complete.json').read_text())
    path = Path(parent['path'])
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != parent['sha256']:
            raise ValueError('Parent source identity changed')
    output = WORK / 'candidates/math-calculus-cas-v1.jsonl'
    if output.exists():
        raise RuntimeError('Existing repair attempt; do not resample or replace')
    counts = collections.Counter()
    pending = []
    with path.open() as incoming, output.open('w', buffering=1) as outgoing, concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context('spawn')) as pool:
        def accept(value):
            if value is None:
                counts['outside_scope_or_not_independently_certified'] += 1
                return
            outgoing.write(json.dumps(value, ensure_ascii=False) + '\n')
            counts['accepted_candidates'] += 1
            counts['tokens'] += value['tokens']
            counts['supervised_tokens'] += value['supervised_tokens']
            counts[value['cas_mode']] += 1
        for line in incoming:
            if time.time() >= args.deadline - 180:
                break
            row = json.loads(line)
            counts['parent_rows_scanned'] += 1
            # Skip non-calculus rows before invoking the parser.
            if not re.search(r'(?i)derivative|differentiate|\\int(?![A-Za-z])', row['prompt']):
                counts['not_calculus'] += 1
                continue
            pending.append(pool.submit(certify, row))
            if len(pending) >= 24:
                accept(pending.pop(0).result())
            if counts['parent_rows_scanned'] % 100 == 0:
                write(JOB / 'math-calculus-cas-v1-status.json', dict(counts=counts, ready_for_training=False))
        for future in pending:
            accept(future.result())
    with output.open('rb') as stream:
        identity = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = dict(path=str(output), bytes=output.stat().st_size, sha256=identity,
                  parent_sha256=parent['sha256'], counts=counts,
                  ready_for_training=False, quality_audit_passed=False,
                  source_repair='computational-calculus-cas-v1',
                  distribution_limitation='Computational univariate calculus only, not the full original math target.',
                  pending=['fixed200 full premise/trace review', 'test and family decontamination', 'native packing'])
    write(JOB / 'math-calculus-cas-v1-complete.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
