"""CPU feasibility pass against immutable old prompts/embedded Python bodies.

The output is a quarantined old-overlap cache, never training-ready data. The
final reviewed new suites, cross-source dedup, source audits and packing remain
mandatory. In particular these counts are upper bounds on final usable tokens.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import numpy as np
from screening import JOB,WORK,norm,code_function_families
from build_ngram_index import spans
sys.path.insert(0,'/workspace/video-model')
from common import tokenizer


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def worker(domain,deadline,completion_path=None,output_path=None):
    completion_path=Path(completion_path) if completion_path else JOB/f'cpt-{domain}-complete.json'
    receipt=json.loads(completion_path.read_text())
    if receipt['domain']!=domain:raise ValueError('Wrong domain completion')
    root=Path(receipt['path']);out=Path(output_path) if output_path else JOB/'cpt-old-screen'/domain
    out.mkdir(parents=True,exist_ok=True)
    if (out/'eligible.idx.u32').exists():raise RuntimeError('Existing screening attempt; inspect rather than duplicate')
    for name,value in receipt['files'].items():
        if (root/name).stat().st_size!=value['bytes'] or sha(root/name)!=value['sha256']:
            raise ValueError('CPT raw source identity changed')
    old_path=WORK/'prior-13word-hashes.npy';old=np.load(old_path,mmap_mode='r')
    identity=dict(source_files=receipt['files'],completion=str(completion_path),completion_sha256=sha(completion_path),prior_ngram_sha256=sha(old_path),
                  code=sha(__file__),screening=sha(JOB/'screening.py'))
    db=None
    if domain=='code':
        complete=JOB/'prior-code-functions-complete.json'
        while not complete.exists() and time.time()<deadline-300:time.sleep(5)
        if not complete.exists():raise TimeoutError('Function exclusion index not complete')
        functions=json.loads(complete.read_text());path=Path(functions['path'])
        if sha(path)!=functions['sha256']:raise ValueError('Function exclusion index changed')
        identity['prior_functions_sha256']=functions['sha256']
        db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
    tok=tokenizer();ids=np.memmap(root/'ids.u16',dtype=np.uint16,mode='r')
    counts=collections.Counter();started=last=time.monotonic()
    with (out/'eligible.idx.u32').open('wb') as eligible, (out/'excluded.jsonl').open('w') as excluded:
        for line in (root/'documents.jsonl').open():
            if time.time()>deadline:raise TimeoutError('Old-overlap screening preparation bound')
            row=json.loads(line);counts['input_documents']+=1;counts['input_tokens']+=row['tokens']
            text=tok.decode(ids[row['offset']:row['offset']+row['tokens']-1].tolist(),skip_special_tokens=False)
            reason=None
            if any(tag in text for tag in ['<|im_start|>','<|im_end|>','<|endoftext|>']):
                reason='literal_chat_control_in_raw_content'
            else:
                values=np.asarray(spans(norm(text)),dtype=np.uint64)
                positions=np.searchsorted(old,values);valid=positions<len(old)
                if (old[positions[valid]]==values[valid]).any():reason='prior_prompt_13word_span'
            if reason is None and db is not None:
                try:
                    keys=code_function_families(text)
                    if any(db.execute('SELECT 1 FROM families WHERE hash=?',(key,)).fetchone() for key in keys):
                        reason='prior_embedded_function_family'
                except Exception:reason='decoded_python_AST_invalid'
            if reason:
                counts[reason+':documents']+=1;counts[reason+':tokens']+=row['tokens']
                excluded.write(json.dumps(dict(index=row['index'],tokens=row['tokens'],reason=reason,text_sha256=row['text_sha256']))+'\n')
            else:
                np.asarray([row['index']],dtype=np.uint32).tofile(eligible)
                counts['old_screen_eligible_documents']+=1;counts['old_screen_eligible_tokens']+=row['tokens']
            if time.monotonic()-last>30:
                write(out/'status.json',dict(domain=domain,counts=counts,ready_for_training=False,
                         elapsed_seconds=time.monotonic()-started,scope='old-overlap only; final usable-token upper bound'))
                last=time.monotonic()
    if db:db.close()
    write(out/'complete.json',dict(domain=domain,counts=counts,identity=identity,
          files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [out/'eligible.idx.u32',out/'excluded.jsonl']},
          elapsed_seconds=time.monotonic()-started,ready_for_training=False,quality_audit_passed=False,
          scope='Old-overlap only. New evaluation exclusion, cross-source dedup, source review and native packing still required.'))
    print(json.dumps(dict(domain=domain,counts=counts,complete=True)),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--domain',choices=['all','math','code','general'],default='all')
    parser.add_argument('--completion');parser.add_argument('--output')
    parser.add_argument('--deadline',type=float,required=True);args=parser.parse_args()
    if bool(args.completion)!=bool(args.output):raise ValueError('Declare both segment completion and independent output')
    if args.domain!='all':worker(args.domain,args.deadline,args.completion,args.output);return
    if args.completion:raise ValueError('Explicit segment screening requires a single domain')
    processes=[]
    for domain in ['math','code','general']:
        stream=(JOB/f'cpt-old-screen-{domain}.log').open('a')
        child=subprocess.Popen([sys.executable,__file__,'--domain',domain,'--deadline',str(args.deadline)],stdout=stream,stderr=subprocess.STDOUT)
        processes.append((domain,child,stream))
    failures=[]
    for domain,child,stream in processes:
        code=child.wait();stream.close()
        if code:failures.append(dict(domain=domain,returncode=code))
    if failures:raise RuntimeError('Bounded old-screen failures: '+json.dumps(failures))


if __name__=='__main__':main()
