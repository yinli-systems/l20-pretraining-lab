"""Collect complete published R1 final solutions, never truncated thinking.

This is the previously approved OpenR1 source option. Selection is the first
source-order generation with complete reasoning, upstream Math-Verify=true,
an independently rechecked final answer, and a complete native-context final
solution. Entire closed <think> sections are omitted; final solutions are never
cut or rewritten. Source-level premise/trace review still gates training.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
import collections
import concurrent.futures
import gc
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import sqlite3
import sys
import time

OLD=Path('/workspace/video-model');JOB=OLD/'validation-48h';WORK=Path('/dev/shm/video-model-validation48')
sys.path.insert(0,str(OLD));sys.path.insert(0,str(JOB/'vendor'))
from screening import digest,family
from numeric_checks import check_solution
TOK=LANG=None


def candidate(row):
    global TOK,LANG
    from common import tokenizer,encode_messages
    from evaluate import math_correct
    from langid.langid import LanguageIdentifier,model
    if TOK is None:TOK=tokenizer()
    if LANG is None:LANG=LanguageIdentifier.from_modelstring(model,norm_probs=True)
    p=row['problem'];answer=row['answer']
    if row.get('question_type')!='math-word-problem' or not answer or re.fullmatch('[A-Ea-e]',answer.strip()):return None
    if '\\boxed' in p or re.search(r'(?i)figure|diagram|shown below|translate.*english|keep.*original.*format|thick lines|red cells|blue cells',p):return None
    flags=row.get('correctness_math_verify') or [];complete=row.get('is_reasoning_complete') or [];finishes=row.get('finish_reasons') or []
    for i,generation in enumerate(row.get('generations') or []):
        if i>=len(flags) or flags[i] is not True or i>=len(complete) or complete[i] is not True:continue
        finish=finishes[i] if i<len(finishes) else None
        if finish not in [None,'stop','eos','eos_token','stop_sequence']:continue
        if '<think>' in generation:
            if generation.count('<think>')!=1 or generation.count('</think>')!=1:continue
            response=generation.split('</think>',1)[1].strip();analysis_omitted=True
        else:response=generation.strip();analysis_omitted=False
        if not response or '\\boxed' not in response or '<think>' in response or re.search(r'\.\.\.$',response):continue
        if any(t in p+response for t in ['<|im_start|>','<|im_end|>','<|endoftext|>']):continue
        if re.search(r'(?i)not enough information|cannot be determined|given.*correct answer|solution.*not provided|option [A-E]',response):continue
        ids,labels=encode_messages(TOK,[dict(role='user',content=p),dict(role='assistant',content=response)])
        n=sum(x!=-100 for x in labels)
        if len(ids)>2048 or not 32<=n<=1000:continue
        language_text=re.sub(r'(?s)\$\$.*?\$\$|\\\[.*?\\\]|\$.*?\$|\\\(.*?\\\)',' ',p+' '+response)
        language,confidence=LANG.classify(language_text)
        if language!='en' or confidence<.9:continue
        checks=check_solution(response)
        if checks['contradictions'] or not math_correct(response,answer):continue
        return dict(source='open-r1/OpenR1-Math-220k',source_config='default',source_id=row['uuid'],
           source_generation_index=i,problem_source=row['source'],prompt=p,response=response,answer=answer,
           prompt_hash=digest(p),family_hash=family(p),tokens=len(ids),supervised_tokens=n,
           published_generation_sha256=hashlib.sha256(generation.encode()).hexdigest(),
           complete_analysis_block_omitted=analysis_omitted,published_final_solution_unchanged=True,
           upstream_math_verify=True,upstream_reasoning_complete=True,upstream_finish_reason=finish,
           finish_reason_semantics='unknown when absent; complete-analysis flag and complete final required, not inferred literal stop',
           independent_numeric_checks=checks['numeric_equalities_checked'],independently_verified_all_premises=False,
           language='en',language_normalized_model_score=float(confidence),review_required=True,
           split_assignment='quarantined_pending_fixed200_review_and_family_decontamination')
    return None


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    from datasets import load_dataset
    p=argparse.ArgumentParser();p.add_argument('--deadline',type=float,required=True);args=p.parse_args()
    spec=json.loads((JOB/'source-access/openr1-math-candidate.json').read_text())
    assert spec['revision']=='e4e141ec9dea9f8326f4d347be56105859b2bd68'
    output=WORK/'candidates/openr1-short.jsonl'
    if output.exists():raise RuntimeError('Existing source attempt, inspect rather than duplicate')
    db=sqlite3.connect('file:'+str(WORK/'prior-prompts.sqlite')+'?mode=ro',uri=True)
    ds=load_dataset(spec['repo'],'default',split='train',revision=spec['revision'],streaming=True)
    iterator=iter(ds);counts=collections.Counter();pending=[];seen=set()
    with output.open('w',buffering=1) as stream,concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        def accept(value):
            if not value:counts['quality_or_context_rejected']+=1;return
            if value['prompt_hash'] in seen:counts['duplicate_prompt']+=1;return
            seen.add(value['prompt_hash']);value['revision']=spec['revision']
            stream.write(json.dumps(value,ensure_ascii=False)+'\n');counts['accepted_candidates']+=1
            counts['tokens']+=value['tokens'];counts['supervised_tokens']+=value['supervised_tokens']
            counts['numeric_equalities_checked']+=value['independent_numeric_checks'];counts['source:'+value['problem_source']]+=1
            counts['complete_analysis_blocks_omitted']+=value['complete_analysis_block_omitted']
        for row in iterator:
            counts['scanned']+=1
            if time.time()>args.deadline-180:break
            h=digest(row['problem']);f=family(row['problem'])
            if h in seen or db.execute('SELECT 1 FROM prompts WHERE hash=? OR family_hash=? LIMIT 1',(h,f)).fetchone():counts['old_prompt_or_family']+=1;continue
            pending.append(pool.submit(candidate,row))
            if len(pending)>=32:accept(pending.pop(0).result())
            if counts['scanned']%1000==0:
                value=dict(status='collecting_complete_short_R1_finals',ready_for_training=False,counts=dict(counts),source=spec)
                write(JOB/'openr1-short-status.json',value);print(json.dumps(value),flush=True)
        for future in pending:accept(future.result())
    iterator.close();db.close();del iterator,ds;gc.collect()
    with output.open('rb') as stream:identity=hashlib.file_digest(stream,'sha256').hexdigest()
    write(JOB/'openr1-short-complete.json',dict(status='quarantined_short_published_finals_collected',ready_for_training=False,
          path=str(output),bytes=output.stat().st_size,sha256=identity,source=spec,counts=dict(counts),
          policy='First source-order qualifying published generation; omit entire closed analysis, retain final unchanged. No new model generation.',
          pending=['fixed200 independent premise/trace review','new evaluation and old-training near/family screening','native packing']))
    print(json.dumps(dict(completed=True,counts=dict(counts),sha256=identity)),flush=True)


if __name__=='__main__':main()
