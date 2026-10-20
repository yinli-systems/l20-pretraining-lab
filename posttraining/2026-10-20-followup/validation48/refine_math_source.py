"""Deterministic repair of language, missing-figure and answer-only classes.

Preserves the failed/mixed and prior candidate files and their fixed samples.
The revised subset still needs its own full source review before any training.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import sys
import time

JOB=Path('/workspace/video-model/validation-48h');WORK=Path('/dev/shm/video-model-validation48')
sys.path.insert(0,str(JOB/'vendor'))
from numeric_checks import check_solution,canaries


def main():
    from langid.langid import LanguageIdentifier,model
    identifier=LanguageIdentifier.from_modelstring(model,norm_probs=True)
    assert identifier.classify('This problem asks for the sum of two integers. We add the numbers to find the answer.')[0]=='en'
    assert identifier.classify('Riješite sustav jednadžbi. Rješenja sustava su uređeni parovi.')[0]!='en'
    p=argparse.ArgumentParser();p.add_argument('--deadline',type=float,required=True);args=p.parse_args()
    prior=json.loads((JOB/'math-olympiads-v3-complete.json').read_text());source=Path(prior['path'])
    with source.open('rb') as stream:identity=hashlib.file_digest(stream,'sha256').hexdigest()
    assert identity==prior['sha256']
    out=WORK/'candidates/math-olympiads-v4.jsonl'
    if out.exists():raise RuntimeError('Existing revised corpus; do not replace or resample')
    counts=collections.Counter();exclusions=JOB/'reviews/math-v4-excluded.jsonl'
    with source.open() as inp,out.open('w') as dest,exclusions.open('w') as reject:
        for i,line in enumerate(inp):
            if time.time()>args.deadline:raise TimeoutError('Preparation deadline during source repair')
            row=json.loads(line);counts['input']+=1;reason=None
            # Colors and underlined/shaded cells are not preserved by plain
            # text scraping. Reject the whole grounded task, not guess colors.
            if re.search(r'(?i)(?:red|blue|green|shaded|colou?red|highlighted) (?:cells?|squares?|regions?|numbers?)',row['prompt']):reason='missing_visual_grounding'
            response=row['response'].split('Final answer:')[0]
            if len(response.split())<45 and not re.search(r'(?i)because|since|therefore|thus|hence|then|reduces|so |we | is | are |\\frac|=',response):reason='answer_only_or_missing_derivation'
            checks=check_solution(response);counts['numeric_equalities_checked']+=checks['numeric_equalities_checked']
            if checks['contradictions']:reason='numeric_identity_contradiction'
            # Remove formulas before language classification. Scores are model
            # confidence, not calibrated probability of source correctness.
            text=row['prompt']+' '+response
            text=re.sub(r'(?s)\$\$.*?\$\$|\\\[.*?\\\]|\$.*?\$|\\\(.*?\\\)',' ',text)
            lang,score=identifier.classify(text)
            if lang!='en' or score<.9:reason='not_confident_English'
            if reason:
                counts[reason]+=1;reject.write(json.dumps(dict(parent_line=i,prompt_hash=row['prompt_hash'],reason=reason,numeric_checks=checks,language=lang,language_score=float(score)))+'\n');continue
            row.update(source_repair_version='olympiads-reference-v4',v3_parent_line=i,
                independent_numeric_checks=checks['numeric_equalities_checked'],language=lang,
                language_normalized_model_score=float(score),review_required=True)
            dest.write(json.dumps(row,ensure_ascii=False)+'\n');counts['accepted_candidates']+=1
            counts['tokens']+=row['tokens'];counts['supervised_tokens']+=row['supervised_tokens']
    with out.open('rb') as stream:identity=hashlib.file_digest(stream,'sha256').hexdigest()
    receipt=dict(status='source_repair_candidates_complete',ready_for_training=False,path=str(out),
      bytes=out.stat().st_size,sha256=identity,parent_sha256=prior['sha256'],counts=dict(counts),
      numeric_canaries=canaries(),language_detector='langid1.1.6 installed only in phase vendor directory',
      remaining_gates=['full fixed200 semantic source review','new evaluation/family decontamination','packing and native mask check'])
    (JOB/'math-olympiads-v4-complete.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
