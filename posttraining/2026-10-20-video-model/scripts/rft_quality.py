"""Conservative training-data checks at the RFT sampling/training handoff."""
import ast
import collections
import hashlib
import json
import re
from pathlib import Path

from common import ROOT, CACHE, tokenizer, write_json, log
from evaluate import extract_code, math_correct

REVISION = 'rft-quality-v2'
# Training examples with independently checked contradictions/incomplete labels.
# These are source-training hashes, never development/final evaluation IDs.
BAD_MATH_PROMPTS = {
    '323418120735577640edf98328e119d9ef3b6fa572e09f365bce1559f90870e3': 'additivity contradicts the supplied f(1), f(2)',
    '14fd765cf7c4d33eb5e74113569f26cead79db33e80ca8cc134115c14abbd8ce': 'quadratic has two valid roots; label arbitrarily excludes one',
    '95d1dc473e10bae6a952f07f7374df72ccc36b4d63196bb68f32e3790f5054bb': 'fraction product equals -40/9, not supplied -16/3',
}


def last_box(text):
    matches = list(re.finditer(r'\\(?:boxed|fbox)\s*\{', text))
    if not matches:
        return None
    start = matches[-1].start()
    opening = matches[-1].end() - 1
    depth = 0
    for i in range(opening, len(text)):
        preceding = 0
        j = i - 1
        while j >= 0 and text[j] == '\\':
            preceding += 1
            j -= 1
        if preceding % 2:
            continue
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1], text[i+1:]
    return None


def short_finished_suffix(suffix):
    remaining = re.sub(r'\\[\]\)]', '', suffix).strip(' \t\r\n$.,;:!`')
    if not remaining:
        return True
    # Permit a short unit/sentence after the answer, not another derivation.
    return (len(suffix) <= 96 and suffix.rstrip().endswith('.')
            and not any(x in suffix for x in ['=', '\\frac', '\\boxed', '\\fbox']))


def rejection_reason(row, tok):
    text = row['completion']
    if len(tok.encode(text, add_special_tokens=False)) >= 1023:
        return 'near_generation_cap'
    if row['source'] == 'math':
        if row['prompt_hash'] in BAD_MATH_PROMPTS:
            return 'confirmed_bad_training_question'
        boxed = last_box(text)
        if boxed is None:
            return 'no_complete_explicit_final_answer'
        if not short_finished_suffix(boxed[1]):
            return 'continuation_after_final_answer'
        if not math_correct(boxed[0], row['answer']):
            return 'explicit_final_answer_fails_existing_math_verify'
        reference_box = last_box(row['reference'])
        if reference_box is None or not math_correct(reference_box[0], row['answer']):
            return 'reference_has_no_matching_explicit_final_answer'
    elif row['source'] == 'code':
        for kind in ['completion', 'reference']:
            if row[kind].count('```') % 2:
                return kind + '_unclosed_code_fence'
            try:
                ast.parse(extract_code(row[kind]))
            except (SyntaxError, ValueError):
                return kind + '_incomplete_python'
    else:
        return 'unexpected_source'
    return None


def filter_rows(rows, tok):
    kept, rejected = [], []
    for row in rows:
        reason = rejection_reason(row, tok)
        if reason:
            rejected.append({'prompt_hash': row['prompt_hash'], 'source': row['source'], 'reason': reason})
        else:
            kept.append(row)
    return kept, rejected


def prepare_quality_dataset(stage, raw_directory):
    roundid = int(stage.split('-')[1])
    source = ROOT / f'rft-{roundid}-accepted.jsonl'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    dest = CACHE / f'rft-{roundid}-{REVISION}'
    marker = ROOT / f'rft-{roundid}-quality-gate.json'
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior.get('revision') == REVISION and prior.get('source_sha256') == source_hash:
            if prior.get('status') == 'insufficient_clean_prompts':
                return None
            if (dest/'manifest.json').exists():
                return dest
    rows = [json.loads(x) for x in source.read_text().splitlines()]
    kept, rejected = filter_rows(rows, tokenizer())
    filtered_path = ROOT / f'rft-{roundid}-quality-accepted.jsonl'
    temp = filtered_path.with_suffix('.jsonl.tmp')
    temp.write_text(''.join(json.dumps(x)+'\n' for x in kept))
    temp.replace(filtered_path)
    record = {'revision': REVISION, 'stage': stage, 'source': str(source),
              'source_sha256': source_hash, 'raw_directory': str(raw_directory),
              'raw_accepted': len(rows), 'clean_accepted': len(kept),
              'source_counts_before': dict(collections.Counter(x['source'] for x in rows)),
              'source_counts_after': dict(collections.Counter(x['source'] for x in kept)),
              'rejection_counts': dict(collections.Counter(x['reason'] for x in rejected)),
              'rejected_items': rejected, 'filtered_rows': str(filtered_path),
              'evaluation_protocol_unchanged': True,
              'limitation': 'Structural/answer checks do not certify all reasoning steps or all question labels.'}
    if len(kept) < 256:
        record['status'] = 'insufficient_clean_prompts'
        write_json(marker, record)
        log('rft_quality_gate_skipped_training', **{k: record[k] for k in ['stage','raw_accepted','clean_accepted']})
        return None
    from rft import make_dataset
    directory, manifest = make_dataset(kept, roundid, directory=dest)
    record.update(status='ready', data=str(directory), manifest=manifest)
    write_json(marker, record)
    log('rft_quality_gate_ready', stage=stage, raw=len(rows), clean=len(kept), rejected=len(rejected))
    return directory
