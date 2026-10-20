"""Source repair after the mixed Numina candidate subset failed review.

Keep the source-tagged olympiads subset, with conservative completeness and
numeric-identity filters. It remains quarantined pending a fresh fixed review.
The failed mixed candidate is immutable and retained as an exclusion input.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import time
from numeric_checks import check_solution,canaries

JOB=Path('/workspace/video-model/validation-48h');WORK=Path('/dev/shm/video-model-validation48')


def main():
    p=argparse.ArgumentParser();p.add_argument('--deadline',type=float,required=True);args=p.parse_args()
    print(json.dumps(canaries()),flush=True)
    prior=json.loads((JOB/'math-replacement-complete.json').read_text());source=Path(prior['path'])
    with source.open('rb') as stream:identity=hashlib.file_digest(stream,'sha256').hexdigest()
    if identity!=prior['sha256']:raise ValueError('Immutable mixed candidate changed')
    out=WORK/'candidates/math-olympiads-v3.jsonl';excluded=JOB/'reviews/math-v3-numeric-excluded.jsonl'
    if out.exists():raise RuntimeError('Existing repaired subset; never resample/rewrite')
    counts=collections.Counter()
    with source.open() as inp,out.open('w') as dest,excluded.open('w') as rejects:
        for i,line in enumerate(inp):
            if time.time()>args.deadline:raise TimeoutError('Math repair preparation deadline')
            row=json.loads(line);counts['input']+=1
            if row['problem_source']!='olympiads':counts['source_excluded']+=1;continue
            if ('\\boxed' in row['prompt'] or re.fullmatch('[A-Ea-e]',row['answer'].strip()) or
                re.search(r'(?i)figure|diagram|shown below|option [A-E]|not provided|steps.*not provided|specific steps',row['prompt']+' '+row['response'])):
                counts['self_containment_or_completeness_excluded']+=1;continue
            result=check_solution(row['response']);counts['numeric_equalities_checked']+=result['numeric_equalities_checked']
            if result['contradictions']:
                counts['numeric_contradiction_excluded']+=1
                rejects.write(json.dumps(dict(source_line=i,prompt_hash=row['prompt_hash'],checks=result),ensure_ascii=False)+'\n');continue
            row.update(source_repair_version='olympiads-reference-v3',parent_line=i,
              independent_numeric_checks=result['numeric_equalities_checked'],
              independently_verified_all_premises=False,review_required=True,
              source_scope='Numina1.5 problem_source=olympiads only; source tag does not itself prove official provenance or proof validity')
            dest.write(json.dumps(row,ensure_ascii=False)+'\n');counts['accepted_candidates']+=1
            counts['tokens']+=row['tokens'];counts['supervised_tokens']+=row['supervised_tokens']
    with out.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    receipt=dict(status='repaired_source_subset_collected',ready_for_training=False,path=str(out),bytes=out.stat().st_size,
        sha256=sha,parent_sha256=identity,counts=dict(counts),numeric_canaries=canaries(),
        pending=['fresh fixed200 source review','source/template near-duplicate audit','new evaluation decontamination','native objective packing'])
    (JOB/'math-olympiads-v3-complete.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
