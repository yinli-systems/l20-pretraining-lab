"""Acquire and CPU-check candidate SFT rows; review/freeze gates remain closed.

Code is independently executed in the unchanged sandbox. Math's answer check is
recorded as answer consistency only, never falsely promoted to premise/proof
verification. All proposed train/evaluation assignments remain quarantined
until source-family/near-duplicate screening and independent review finish.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import ast
import collections
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import time

OLD = Path('/workspace/video-model')
JOB = OLD/'validation-48h'
WORK = Path('/dev/shm/video-model-validation48')
sys.path.insert(0,str(OLD))
from common import tokenizer, encode_messages
from prepare import digest as prompt_digest, norm


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def family_digest(prompt):
    return hashlib.sha256(re.sub(r'\d+','#',norm(prompt)).encode()).hexdigest()


def verify_code(row):
    from evaluate import extract_code, code_correct
    try:
        tests=json.loads(row['unit_tests']) if isinstance(row['unit_tests'],str) else row['unit_tests']
        if len(tests)<3:
            return None
        code=extract_code(row['output'])
        tree=ast.parse(code)
        functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        test_tree=ast.parse('\n'.join(tests))
        names=collections.Counter(n.func.id for n in ast.walk(test_tree)
                 if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in functions)
        if not names:
            return None
        entry=names.most_common(1)[0][0]
        # Commit to the frozen first-entrypoint block before executing tests.
        code=extract_code(row['output'],entry_point=entry)
        tree=ast.parse(code)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==entry)
        calls=[n for n in ast.walk(test_tree) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Name) and n.func.id==entry]
        if len({ast.dump(n) for n in calls})<3:
            return None
        # Require at least an explicit empty/zero/negative boundary fixture.
        boundary=any(isinstance(n,(ast.List,ast.Tuple,ast.Set)) and not n.elts
                     or isinstance(n,ast.Dict) and not n.keys
                     or isinstance(n,ast.Constant) and n.value in [0,'']
                     or isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub)
                     for call in calls for n in ast.walk(call))
        if not boundary:
            return None
        test='\n'.join(tests)
        if not code_correct(code,test):
            return None
        if code_correct('',test):
            return None
        for wrong in ['None','0','[]']:
            if code_correct(f'def {entry}(*args, **kwargs): return {wrong}',test):
                return None
        skeleton=ast.dump(tree,include_attributes=False)
        # Provisional source-family exclusion, followed by the independent
        # stronger near-duplicate/reference-code review gate before packing.
        fam=family_digest(row['input'])
        return dict(source='nvidia/OpenCodeInstruct',source_id=row['id'],prompt=row['input'],
                    response=row['output'],prompt_hash=prompt_digest(row['input']),family_hash=fam,
                    reference=code,test=test,entry_point=entry,
                    verification='all supplied tests replayed; >=3 distinct calls with boundary; empty/None/zero/empty-list controls rejected',
                    independently_replayed_code=True,reference_ast_sha256=hashlib.sha256(skeleton.encode()).hexdigest())
    except Exception:
        return None


def check_row(mode,row,tok):
    if mode=='math':
        from evaluate import math_correct
        p=row['problem'];answer=row.get('expected_answer','');response=row['generated_solution']
        if not answer or '\\boxed' not in response or '<think>' in response and '</think>' not in response:
            return None
        if not math_correct(response,answer):
            return None
        return dict(source='nvidia/OpenMathInstruct-2',prompt=p,response=response,
                    answer=answer,prompt_hash=prompt_digest(p),family_hash=family_digest(p),
                    verification='published expected answer agrees with frozen math-verify; independent premise/solution review pending',
                    independently_verified_premises=False,problem_source=row.get('problem_source'))
    messages=row.get('messages',[])
    if not messages or messages[-1]['role']!='assistant':
        return None
    p='\n'.join(m['content'] for m in messages if m['role']=='user')
    return dict(source='HuggingFaceTB/smoltalk',prompt=p,response=messages[-1]['content'],messages=messages,
                prompt_hash=prompt_digest(p),family_hash=family_digest(p),verification='published general replay; source review pending')


def worker(mode,deadline,resume=False):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    access=json.loads((JOB/'source-access'/f'sft-{mode if mode!="general" else "general"}.json').read_text())
    tok=tokenizer()
    db=sqlite3.connect('file:'+str(WORK/'prior-prompts.sqlite')+'?mode=ro',uri=True)
    ds=load_dataset(access['repo'],access['config'],split=access['split'],revision=access['revision'],streaming=True)
    output=WORK/'candidates'/f'{mode}.jsonl'
    output.parent.mkdir(exist_ok=True)
    if output.exists() and not resume:
        raise RuntimeError('Existing candidate output; explicit continuation cursor required, do not duplicate source scans')
    counts=collections.Counter()
    seen=set()
    previous=None
    if resume:
        record_path=JOB/f'candidates-{mode}-complete.json'
        if not record_path.exists():
            record_path=JOB/f'candidates-{mode}-interrupted.json'
        previous=json.loads(record_path.read_text())
        with output.open('rb') as stream:
            prior_sha=hashlib.file_digest(stream,'sha256').hexdigest()
        if previous['source_revision']!=access['revision'] or previous['sha256']!=prior_sha:
            raise RuntimeError('Candidate continuation source/content identity mismatch')
        counts.update(previous['counts'])
        seen={json.loads(line)['prompt_hash'] for line in output.read_text().splitlines()}
        # The original worker counts the row that triggered its bound before
        # breaking. Conservatively revisit that one row, with exact dedup.
        # An outer CPU timeout can interrupt before pending checks drain. Its
        # explicitly audited prefix checkpoint replays from zero, with exact
        # accepted-prompt dedup, rather than guessing an unsafe source cursor.
        skip=0 if previous.get('status')=='interrupted_prefix_checkpoint' else max(0,counts['scanned']-1)
        counts['scanned']=skip
        ds=ds.skip(skip)
    targets=dict(math=100000,code=101200,general=30000)
    limit=dict(math=1500000,code=2500000,general=150000)[mode]
    pool=concurrent.futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn')) if mode=='code' else None
    pending=[]
    with output.open('a' if resume else 'w',buffering=1) as stream:
        def accept(record):
            if not record:
                counts['validation_rejected']+=1
                return
            p=record['prompt'];h=record['prompt_hash'];fam=record['family_hash']
            if h in seen:
                counts['duplicates']+=1
                return
            seen.add(h)
            messages=record.get('messages') or [dict(role='user',content=p),dict(role='assistant',content=record['response'])]
            ids,labels=encode_messages(tok,messages)
            answer_tokens=sum(x!=-100 for x in labels)
            if len(ids)>2048 or not 200<=answer_tokens<=1000:
                counts['length_rejected']+=1
                return
            record.update(revision=access['revision'],tokens=len(ids),supervised_tokens=answer_tokens,
                          finish_reason='published complete response; no local generation',
                          split_assignment='quarantined_not_yet_training_or_independent_evaluation',
                          additional_review_required=True)
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            counts['accepted_candidates']+=1
            counts['nonpadding_tokens']+=len(ids)
            counts['supervised_tokens']+=answer_tokens
        try:
            for row in ds:
                counts['scanned']+=1
                if time.time()>=deadline-120 or counts['accepted_candidates']>=targets[mode] or counts['scanned']>=limit:
                    break
                p=row.get('input',row.get('problem',''))
                if mode=='general':
                    p='\n'.join(m['content'] for m in row.get('messages',[]) if m['role']=='user')
                if not p or len(p)>6500:
                    continue
                h=prompt_digest(p);fam=family_digest(p)
                if h in seen:
                    counts['duplicates']+=1
                    continue
                if db.execute('SELECT 1 FROM prompts WHERE hash=? OR family_hash=? LIMIT 1',(h,fam)).fetchone():
                    counts['old_prompt_or_numeric_family_rejected']+=1
                    continue
                if mode=='code':
                    if len(row.get('output',''))>10000:
                        continue
                    pending.append(pool.submit(verify_code,row))
                    if len(pending)>=36:
                        accept(pending.pop(0).result())
                else:
                    accept(check_row(mode,row,tok))
                if counts['scanned']%1000==0:
                    value=dict(mode=mode,status='building_quarantined_candidates',counts=dict(counts),ready_for_training=False)
                    write(JOB/f'candidates-{mode}-status.json',value)
                    print(json.dumps(value),flush=True)
            for future in pending:
                accept(future.result())
        finally:
            if pool:
                pool.shutdown(cancel_futures=True)
    db.close()
    with output.open('rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    write(JOB/f'candidates-{mode}-complete.json',dict(mode=mode,status='candidate_collection_complete',
          counts=dict(counts),path=str(output),bytes=output.stat().st_size,sha256=digest,ready_for_training=False,
          source_revision=access['revision'],pending=['cross-source family and near-duplicate grouping','200-example source review','final math premise/solution verification','new suite freeze and corpus packing']))
    if previous:
        write(JOB/f'candidates-{mode}-continuation.json',dict(previous_sha256=previous['sha256'],
               previous_bytes=previous['bytes'],previous_counts=previous['counts'],new_sha256=digest,
               source_revision=access['revision'],new_counts=dict(counts)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['all','math','code','general'],default='all')
    parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.mode!='all':
        worker(args.mode,args.deadline,args.resume)
        return
    while not (JOB/'prior-prompt-index.json').exists() and time.time()<args.deadline-300:
        time.sleep(5)
    if not (JOB/'prior-prompt-index.json').exists():
        raise RuntimeError('Previous prompt index unavailable before preparation bound')
    jobs=[]
    for mode in ['math','code','general']:
        log=(JOB/f'candidates-{mode}.log').open('a')
        job=subprocess.Popen([sys.executable,__file__,'--mode',mode,'--deadline',str(args.deadline)],stdout=log,stderr=subprocess.STDOUT)
        jobs.append((mode,job,log))
    failed=[]
    for mode,job,log in jobs:
        code=job.wait();log.close()
        if code:
            failed.append(dict(mode=mode,returncode=code))
    write(JOB/'candidate-collection.json',dict(status='failed' if failed else 'complete',failures=failed,ready_for_training=False))
    if failed:
        raise RuntimeError('Candidate collection requires inspection')


if __name__=='__main__':
    main()
