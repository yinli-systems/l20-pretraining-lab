"""Deterministic, unapproved whitelist from an exhaustive recorded math review.

This never repairs an answer or edits a published trace. All errata stay in the
original corpus and exclusion ledger. A full source review is not a training
quality seal: global overlap, suite identity and production packing still apply.
"""
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def computational_family(row):
    tree = ast.parse(row['independently_parsed_task'], mode='eval')

    class Shape(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                if node.func.id in ['Integer', 'Rational', 'Float']:
                    return ast.Name(id='NUMERIC_PARAMETER', ctx=ast.Load())
                if node.func.id == 'Symbol':
                    return ast.Name(id='VARIABLE', ctx=ast.Load())
            return self.generic_visit(node)

    shape = ast.dump(Shape().visit(tree), include_attributes=False)
    return hashlib.sha256((row['cas_mode'] + ':' + shape).encode()).hexdigest()


def curate(source, review_paths, output, report_path):
    reviews = [json.loads(Path(path).read_text()) for path in review_paths]
    decisions = {}
    for review in reviews:
        if not review.get('review_complete'):
            raise ValueError('An exhaustive review packet is incomplete')
        if review['source_sha256'] != sha(source):
            raise ValueError('Review belongs to a different original source')
        for row in review['rows']:
            if row['prompt_hash'] in decisions:
                raise ValueError('Duplicated review decision')
            decisions[row['prompt_hash']] = row
    rows = [json.loads(line) for line in Path(source).read_text().splitlines()]
    if len({row['prompt_hash'] for row in rows}) != len(rows):
        raise ValueError('Original source has repeated prompts')
    if set(decisions) != {row['prompt_hash'] for row in rows}:
        raise ValueError('Review does not exhaust the original source')
    if output.exists() or report_path.exists():
        raise RuntimeError('Existing whitelist is immutable; inspect it')
    # Sorting/performance feedback never chooses a trace. Keep the first source
    # occurrence with a consistent review; coarsely group structural numerical
    # variants in addition to text-based family keys.
    allowed = {'fully_read_consistent_no_global_quality_claim',
               'template_family_grouping_required'}
    kept, excluded, seen_text, seen_shape = [], [], set(), set()
    counts = collections.Counter()
    for i, row in enumerate(rows):
        decision = decisions[row['prompt_hash']]
        reason = None
        if decision['verdict'] not in allowed:
            reason = decision['verdict']
        elif row['response'].count('```') % 2:
            reason = 'unbalanced_code_fence'
        elif not row.get('strict_full_latex_parse'):
            reason = 'missing_original_full_parse'
        shape = computational_family(row)
        if reason is None and (row['family_hash'] in seen_text or shape in seen_shape):
            reason = 'conservative_computational_or_text_template_duplicate'
        if reason:
            counts[reason] += 1
            excluded.append(dict(source_line=i, prompt_hash=row['prompt_hash'],
                                 reason=reason, review=decision,
                                 original_record=row))
            continue
        seen_text.add(row['family_hash'])
        seen_shape.add(shape)
        kept.append(dict(row, exhaustive_manual_review=decision,
                         conservative_computational_family=shape,
                         original_source_line=i,
                         split_assignment='quarantined_reviewed_whitelist_pending_global_screen',
                         review_required=False,
                         quality_audit_passed=False))
    sys.path.insert(0, '/workspace/video-model')
    from common import tokenizer, encode_messages
    tok = tokenizer()
    tokens = supervised = 0
    for row in kept:
        ids, labels = encode_messages(tok, [dict(role='user', content=row['prompt']),
                                           dict(role='assistant', content=row['response'])])
        n = sum(label != -100 for label in labels)
        if len(ids) != row['tokens'] or n != row['supervised_tokens']:
            raise ValueError('Original native token counts do not replay')
        if len(ids) > 2048 or not 32 <= n <= 1000:
            raise ValueError('Reviewed trace fails the native complete-sample bound')
        tokens += len(ids)
        supervised += n
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger = output.with_suffix('.excluded.jsonl')
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in kept))
    ledger.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in excluded))
    report = dict(source=str(source), source_sha256=sha(source), original_rows=len(rows),
                  exhaustive_review_rows=len(decisions), retained_rows=len(kept),
                  distinct_structural_families=len(seen_shape),
                  native_tokens=tokens, supervised_tokens=supervised,
                  exclusions=dict(counts), quality_audit_passed=False,
                  training_use_allowed=False, semantic_review_complete=True,
                  original_published_prompt_and_response_unchanged=True,
                  reviews={str(path):sha(path) for path in review_paths},
                  files={str(path):dict(bytes=path.stat().st_size, sha256=sha(path))
                         for path in [output, ledger]},
                  code_sha256=sha(__file__),
                  remaining=['fresh suite freeze and global test/old data screening',
                             'source provenance/license seal', 'whole-objective packing/replay',
                             'honest sparse-data feasibility and full backup capacity'],
                  limitation='Human-model semantic review plus CAS controls is not a formal proof of every inference. Conservative structural grouping may exclude distinct valid tasks. This sparse corpus is not the original 100k-question goal.')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--reviews', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(curate(args.source, args.reviews, args.output, args.report),
                     ensure_ascii=False, indent=2))
