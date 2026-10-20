"""A separately audited repair with strict CAS parsing and exact task solving.

No model is generated or chosen against evaluation tests. Failed broad-source
rows are admitted here only after independent task computation; full published
final explanations still require the separate fixed200 semantic source audit.
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
import certify_calculus as calculus
sys.path.insert(0, str(calculus.JOB / 'vendor'))
JOB, WORK = calculus.JOB, calculus.WORK


def reject_ambiguous_mixed_number(text):
    # ANTLR interprets10\\frac{1}{2} as multiplication, while a school
    # arithmetic question normally intends the mixed number10+1/2. A task and
    # label parsed the same wrong way can spuriously agree. Exclude this syntax
    # from automatic certificates; manual semantic validation is separate.
    if re.search(r'(?<![A-Za-z0-9])\d+\s*\\frac\s*\{\s*\d+\s*\}\s*\{\s*\d+\s*\}',text):
        raise ValueError('Mixed-number versus implicit-product ambiguity')


def certify(row):
    import sympy as sp
    from sympy.parsing.latex import parse_latex
    signal.signal(signal.SIGALRM, calculus.expired)
    signal.alarm(8)
    try:
        prompt, final = row['prompt'], row['response']
        if re.search(r'(?i)prove|show that|maximum|minimum|smallest|largest|integer|natural number|parameter|partial|second derivative|likely|intended|typo|complex|positive root|negative root|interval', prompt + ' ' + final):
            return None
        expression = calculus.expressions(prompt)
        if not expression or len(expression) > 400:
            return None
        reject_ambiguous_mixed_number(expression)
        reject_ambiguous_mixed_number(row['answer'])
        task = parse_latex(expression, strict=True)
        answer = parse_latex(row['answer'], strict=True)
        x = sp.Symbol('x')
        if re.search(r'(?i)derivative|differentiate|\\int(?![A-Za-z])', prompt):
            # Strict preparse prevents a valid prefix plus ignored trailing
            # symbols from reaching the original calculus certificate.
            value = calculus.certify(row)
            if value is None:
                return None
            mode = value['cas_mode']
        elif isinstance(task, sp.Equality) and re.search(r'(?i)solve|roots of|solutions of', prompt):
            if task.free_symbols != {x}:
                return None
            polynomial = sp.Poly(task.lhs - task.rhs, x)
            if not 1 <= polynomial.degree() <= 4:
                return None
            roots = sp.solveset(polynomial.as_expr(), x, domain=sp.S.Reals)
            # Multi-root answer syntax is deliberately unsupported here.
            if not isinstance(roots, sp.FiniteSet) or len(roots) != 1:
                return None
            if isinstance(answer, sp.Equality) and answer.lhs == x:
                answer = answer.rhs
            if answer.free_symbols or sp.simplify(next(iter(roots)) - answer) != 0:
                return None
            mode = 'single-real-root-polynomial-independent-exact-solve'
            value = dict(row)
        elif not task.free_symbols and not isinstance(task, sp.Equality) and re.search(r'(?i)calculate|evaluate|simplify|value of|compute', prompt):
            # A lone numerical expression is the entire computational task;
            # geometric/word-problem premises cannot be ignored to certify it.
            if re.search(r'(?i)triangle|circle|polygon|sphere|rectangle|angle|length|area|volume|let |given|if |where|such that', prompt):
                return None
            exact = task.doit()
            if answer.free_symbols or exact.has(sp.Integral, sp.Limit, sp.Sum, sp.oo, -sp.oo, sp.zoo, sp.nan):
                return None
            if sp.simplify(exact - answer) != 0:
                return None
            mode = 'numeric-expression-independent-exact-computation'
            value = dict(row)
        else:
            return None
        value.update(source_repair='computational-symbolic-v2', cas_mode=mode,
                     independently_parsed_task=sp.srepr(task), strict_full_latex_parse=True,
                     independent_computational_gold_verified=True, all_reasoning_steps_certified=False,
                     sympy_version=sp.__version__,
                     split_assignment='quarantined_pending_fixed200_semantic_review_and_decontamination')
        return value
    except Exception:
        return None
    finally:
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    parent = json.loads((JOB / 'openr1-short-complete.json').read_text())
    path = Path(parent['path'])
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != parent['sha256']:
            raise ValueError('Parent source identity changed')
    output = WORK / 'candidates/math-computational-v2.jsonl'
    if output.exists():
        raise RuntimeError('Existing repair attempt; do not duplicate or resample')
    counts, pending = collections.Counter(), []
    with path.open() as incoming, output.open('w', buffering=1) as outgoing, concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context('spawn')) as pool:
        def accept(value):
            if value is None:
                counts['not_independently_certified'] += 1
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
            if not re.search(r'(?i)derivative|differentiate|\\int(?![A-Za-z])|solve|roots of|solutions of|calculate|evaluate|simplify|value of|compute', row['prompt']):
                counts['not_computational_candidate'] += 1
                continue
            pending.append(pool.submit(certify, row))
            if len(pending) >= 24:
                accept(pending.pop(0).result())
            if counts['parent_rows_scanned'] % 100 == 0:
                calculus.write(JOB / 'math-computational-v2-status.json', dict(counts=counts, ready_for_training=False))
        for future in pending:
            accept(future.result())
    with output.open('rb') as stream:
        identity = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = dict(path=str(output), bytes=output.stat().st_size, sha256=identity,
                  parent_sha256=parent['sha256'], source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  counts=counts, ready_for_training=False, quality_audit_passed=False,
                  source_repair='computational-symbolic-v2',
                  distribution_limitation='Only independently solved numerical, single-real-root polynomial and univariate calculus tasks. Unique volume is measured, not filled by repetition.',
                  pending=['fixed200 full premise/trace review', 'test and family decontamination', 'native packing'])
    calculus.write(JOB / 'math-computational-v2-complete.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
