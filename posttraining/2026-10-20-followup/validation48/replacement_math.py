"""Collect real-source math references after augmented-math audit failure.

Upstream validity flags are filters, not independent proof. Numeric equations
are separately recomputed where unambiguous; a new 200-row semantic audit and
full evaluation decontamination still gate all training use.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import collections
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import sqlite3
import sys
import time

OLD=Path('/workspace/video-model');JOB=OLD/'validation-48h';WORK=Path('/dev/shm/video-model-validation48')
sys.path.insert(0,str(OLD))
from common import tokenizer,encode_messages
from screening import digest,family


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def validate(row):
    from evaluate import math_correct
    from sympy.parsing.latex import parse_latex
    from sympy import simplify
    p,a,solution=row['problem'],row['answer'],row['solution']
    if re.search(r'(?:\n|^)[1-3][.)]|\(1\).*\(2\)',p,re.S):return None
    if re.search(r'!\[|\\includegraphics|https?://|\\begin\{array\}',p+solution):return None
    if len(re.findall(r'[\u3400-\u9fff]',p+solution))>4:return None
    if re.search(r'(?i)not enough information|cannot (?:be determined|determine)|apolog|incorrect (?:problem|question)|no solution.*provided',solution):return None
    tail=solution[solution.rfind('\\boxed'):] if '\\boxed' in solution else solution[-600:]
    if not math_correct(tail,a):return None
    equations=re.findall(r'(?s)\$\$(.*?)\$\$|\\\[(.*?)\\\]|(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)',solution)
    checked=0
    for values in equations:
        text=next((v for v in values if v),'').strip()
        if text.count('=')!=1 or re.search(r'[<>]|\\(?:le|ge|approx|Rightarrow|Leftrightarrow|begin|text)',text):continue
        bare=re.sub(r'\\[A-Za-z]+','',text)
        if re.search('[A-Za-z]',bare) or len(text)>250:continue
        try:
            left,right=map(parse_latex,text.split('='))
            if left.free_symbols or right.free_symbols or not left.is_number or not right.is_number:continue
            difference=simplify(left-right)
            if difference!=0 and abs(float(difference.evalf()))>1e-6*max(1,abs(float(left.evalf())),abs(float(right.evalf()))):return None
            checked+=1
        except Exception:continue
        if checked==4:break
    if '\\boxed' not in solution:solution=solution+'\n\nFinal answer: \\boxed{'+a+'}.'
    return dict(source='AI-MO/NuminaMath-1.5',problem_source=row['source'],prompt=p,response=solution,
         answer=a,problem_type=row['problem_type'],prompt_hash=digest(p),family_hash=family(p),
         numeric_equations_independently_recomputed=checked,
         verification='Non-synthetic source reference, upstream problem/solution valid Yes; final answer consistency plus restricted numeric equation checks. Independent source200 review still required.',
         independently_verified_all_premises=False,review_required=True,source_validity_flags={'problem':'Yes','solution':'Yes'})


def main():
    from datasets import load_dataset
    p=argparse.ArgumentParser();p.add_argument('--deadline',type=float,required=True);args=p.parse_args()
    spec=json.loads((JOB/'source-access/math-replacement.json').read_text());tok=tokenizer()
    out=WORK/'candidates/math-replacement.jsonl'
    if out.exists():raise RuntimeError('Existing replacement corpus; inspect before continuation')
    db=sqlite3.connect('file:'+str(WORK/'prior-prompts.sqlite')+'?mode=ro',uri=True)
    counts=collections.Counter();seen=set();pending=[]
    ds=load_dataset(spec['repo'],split='train',revision=spec['revision'],streaming=True)
    with out.open('w',buffering=1) as stream,concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        def accept(record):
            if record is None:counts['verification_rejected']+=1;return
            ids,labels=encode_messages(tok,[dict(role='user',content=record['prompt']),dict(role='assistant',content=record['response'])])
            supervised=sum(x!=-100 for x in labels)
            if len(ids)>2048 or not 32<=supervised<=1000:counts['length_rejected']+=1;return
            if record['prompt_hash'] in seen:counts['duplicate_prompt']+=1;return
            seen.add(record['prompt_hash'])
            record.update(revision=spec['revision'],tokens=len(ids),supervised_tokens=supervised,
              split_assignment='quarantined_pending_independent_review_and_decontamination',
              short_solution_policy='Complete32-1000tokens; do not pad simple correct solutions to200tokens. No truncation.')
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            counts['accepted_candidates']+=1;counts['tokens']+=len(ids);counts['supervised_tokens']+=supervised
            counts['numeric_equations_checked']+=record['numeric_equations_independently_recomputed']
            counts['source:'+record['problem_source']]+=1
        for row in ds:
            counts['scanned']+=1
            if time.time()>args.deadline-120 or counts['accepted_candidates']>=100000:break
            if (row.get('synthetic') or row.get('problem_is_valid')!='Yes' or row.get('solution_is_valid')!='Yes'
                or row.get('question_type')!='math-word-problem' or row.get('source') not in ['cn_k12','olympiads','aops_forum','math','amc_aime']
                or not row.get('answer') or row['answer'] in ['proof','notfound'] or not row.get('solution')):
                counts['metadata_rejected']+=1;continue
            h=digest(row['problem']);f=family(row['problem'])
            if h in seen or db.execute('SELECT 1 FROM prompts WHERE hash=? OR family_hash=? LIMIT 1',(h,f)).fetchone():counts['old_or_duplicate']+=1;continue
            pending.append(pool.submit(validate,row))
            if len(pending)>=32:accept(pending.pop(0).result())
            if counts['scanned']%1000==0:
                value=dict(status='collecting_replacement_math',ready_for_training=False,source=spec,counts=dict(counts))
                write(JOB/'math-replacement-status.json',value);print(json.dumps(value),flush=True)
        for future in pending:accept(future.result())
    db.close()
    with out.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    write(JOB/'math-replacement-complete.json',dict(status='replacement_candidate_collection_complete',ready_for_training=False,
         source=spec,counts=dict(counts),path=str(out),bytes=out.stat().st_size,sha256=sha,
         rejected_original='OpenMath augmented candidate subset failed predeclared source audit',
         pending=['new source200 semantic review','evaluation/family decontamination','source mixture and packing']))


if __name__=='__main__':main()
