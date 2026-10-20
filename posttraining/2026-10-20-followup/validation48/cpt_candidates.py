"""Quarantined, deterministic CPU acquisition of actual domain-CPT tokens.

No training-ready marker is written here. A later pass screens every document
against the frozen new suite and packs the measured token mixture. Candidate
content/metadata, including individual Python licenses, is hashed and retained.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import ast
import collections
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlparse

OLD=Path('/workspace/video-model')
JOB=OLD/'validation-48h'
WORK=Path('/dev/shm/video-model-validation48')
sys.path.insert(0,str(OLD))
from common import tokenizer

TARGET={'math':340_000_000,'code':340_000_000,'general':170_000_000}
MATH_DOMAINS={'wikipedia.org','math.stackexchange.com','mathoverflow.net','artofproblemsolving.com',
    'math.libretexts.org','khanacademy.org','proofwiki.org','planetmath.org','mathworld.wolfram.com',
    'brilliant.org','cut-the-knot.org','mathpages.com','tutorial.math.lamar.edu','ocw.mit.edu',
    'openstax.org','mathsisfun.com','nrich.maths.org','mathcentre.ac.uk','ams.org'}


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def english(text):
    cjk=len(re.findall(r'[\u3400-\u9fff]',text))
    return cjk<max(4,len(text)*.005) and len(re.findall('[A-Za-z]',text))>len(text)*.3


def content(row):
    import requests
    blob=row['blob_id']
    url='https://softwareheritage.s3.amazonaws.com/content/'+blob
    try:
        for attempt in range(3):
            response=requests.get(url,timeout=(10,20))
            if response.status_code==200:
                raw=gzip.decompress(response.content)
                break
            if response.status_code not in [429,500,502,503,504]:
                return None
            time.sleep(.5*(attempt+1))
        else:
            return None
        if len(raw)!=row['length_bytes']:
            return None
        raw_sha1=hashlib.sha1(raw).hexdigest()
        git_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob not in [raw_sha1,git_sha1]:
            return None
        text=raw.decode(row['src_encoding'] or 'utf-8')
        tree=ast.parse(text)
        defs=sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) for n in ast.walk(tree))
        if defs<2:
            return None
        return text,dict(blob_id=blob,blob_hash_semantics='raw-sha1' if blob==raw_sha1 else 'git-blob-sha1',
             repo_name=row['repo_name'],path=row['path'],licenses=row['detected_licenses'],
             license_type=row['license_type'],score=row['score'],source_byte_sha256=hashlib.sha256(raw).hexdigest())
    except Exception:
        return None


def acquire(domain,deadline,topup_tokens=None):
    import numpy as np
    from datasets import load_dataset
    spec=json.loads((JOB/'source-access'/f'cpt-{domain}.json').read_text())
    target=TARGET[domain] if topup_tokens is None else topup_tokens
    parent=None
    if topup_tokens is not None:
        if domain not in ['math','code'] or not 0<target<=200_000_000:
            raise ValueError('Bounded math/code raw-content top-up only')
        parent_path=JOB/f'cpt-{domain}-complete.json'
        parent=json.loads(parent_path.read_text())
        for name,value in parent['files'].items():
            path=Path(parent['path'])/name
            with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if path.stat().st_size!=value['bytes'] or digest!=value['sha256']:
                raise ValueError('Original completed CPT source changed')
        if parent['source']!=spec:
            raise ValueError('Top-up must preserve the pinned source specification')
    suffix='' if parent is None else '-topup-v1'
    out=WORK/'cpt-candidates'/(domain+suffix)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'ids.u16').exists():
        raise RuntimeError('Existing corpus; inspect source cursor before another acquisition')
    tok=tokenizer()
    if tok.convert_tokens_to_ids('<|endoftext|>')!=0:
        raise RuntimeError('Unexpected native document separator')
    data=load_dataset(spec['repo'],spec['config'],split=spec['split'],revision=spec['revision'],streaming=True)
    # The original collector may have discarded an in-flight tail when its
    # token buffer filled. Starting after its observed scan cursor preserves
    # the original bytes and never reuses that prefix; it does not claim an
    # exhaustive traversal or recover unrecorded pending records.
    skip=0 if parent is None else parent['counts']['scanned']
    if skip:data=data.skip(skip)
    prefix=f'cpt-{domain}{suffix}'
    counts=collections.Counter();seen=set();offsets=[0];batch=[]
    started=time.monotonic();last=started
    ids=(out/'ids.u16').open('wb');meta=(out/'documents.jsonl').open('w',buffering=1)
    def flush():
        nonlocal last
        if not batch:return
        encoded=tok([x[0] for x in batch],add_special_tokens=False,return_attention_mask=False)['input_ids']
        for (text,record),tokens in zip(batch,encoded):
            if len(tokens)<256 or len(tokens)>100_000:
                counts['token_length_rejected']+=1;continue
            if counts['tokens']>=target:break
            tokens=tokens+[0]
            np.asarray(tokens,dtype=np.uint16).tofile(ids)
            record.update(index=counts['documents'],offset=offsets[-1],tokens=len(tokens),
                normalized_text_sha256=hashlib.sha256(' '.join(text.split()).encode()).hexdigest(),
                text_sha256=hashlib.sha256(text.encode()).hexdigest())
            offsets.append(offsets[-1]+len(tokens));meta.write(json.dumps(record,ensure_ascii=False)+'\n')
            counts['documents']+=1;counts['tokens']+=len(tokens)
        batch.clear()
        if time.monotonic()-last>20:
            value=dict(status='collecting_quarantined_content',domain=domain,counts=dict(counts),
                       target_nonpadding_tokens=target,ready_for_training=False,
                       measured_tokens_per_second=counts['tokens']/max(1,time.monotonic()-started))
            write(JOB/f'{prefix}-status.json',value);print(json.dumps(value),flush=True);last=time.monotonic()
    def add(item):
        if item is None:counts['content_failed_or_invalid']+=1;return
        text,record=item
        text_hash=hashlib.sha256(text.encode()).hexdigest()
        if text_hash in seen:counts['duplicate_document']+=1;return
        seen.add(text_hash)
        if domain!='code' and not english(text):counts['language_rejected']+=1;return
        batch.append((text,record))
        if len(batch)>=32:flush()
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=24) if domain=='code' else None
    pending=[]
    try:
        for row in data:
            counts['scanned']+=1
            if counts['tokens']>=target or time.time()>deadline-120:break
            if domain=='code':
                # Audit individual files: the dataset card alone supplies no
                # license, and its first samples contain unlicensed material.
                if (row.get('license_type')!='permissive' or not row.get('detected_licenses')
                    or not 4096<=row.get('length_bytes',0)<=262144 or row.get('score',0)<3):
                    counts['metadata_quality_or_license_rejected']+=1;continue
                pending.append(pool.submit(content,row))
                if len(pending)>=72:add(pending.pop(0).result())
            else:
                text=row.get('text','')
                if not 800<=len(text)<=250_000:
                    counts['text_length_rejected']+=1;continue
                if domain=='math':
                    host=(urlparse(row.get('url','')).hostname or '').lower()
                    allowed=any(host==x or host.endswith('.'+x) for x in MATH_DOMAINS)
                    metadata=json.loads(row.get('metadata') or '{}')
                    if not allowed or not metadata.get('found_math') or row.get('int_score',0)<4:
                        counts['math_domain_or_content_quality_rejected']+=1;continue
                    # Explicitly exclude essay-spam redirects observed in the
                    # initial sample despite its nominal education score.
                    if re.search(r'masterpapers|write.{0,15}paper online|buy.{0,12}essay',text,re.I):
                        counts['spam_rejected']+=1;continue
                elif row.get('int_score',0)<3 or row.get('language_score',0)<.95:
                    counts['general_quality_rejected']+=1;continue
                add((text,dict(url=row.get('url'),source_id=row.get('id'),score=row.get('score'),license=spec['license_metadata'])))
        for future in pending:add(future.result())
        flush()
    finally:
        if pool:pool.shutdown(cancel_futures=True)
        ids.close();meta.close()
    np.save(out/'offsets.npy',np.asarray(offsets,dtype=np.int64))
    files={}
    for path in out.iterdir():
        with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        files[path.name]=dict(bytes=path.stat().st_size,sha256=sha)
    write(JOB/f'{prefix}-complete.json',dict(status='quarantined_content_collected',domain=domain,
          counts=dict(counts),target_met=counts['tokens']>=target,ready_for_training=False,
          source=spec,files=files,path=str(out),seconds=time.monotonic()-started,
          target_nonpadding_tokens=target,skipped_source_prefix_rows=skip,
          parent_complete_sha256=None if parent is None else hashlib.sha256(parent_path.read_bytes()).hexdigest(),
          remaining_gates=['new evaluation decontamination','cross-domain exact/near duplicate screen','200-document source review','measured mixture packing']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--domain',choices=['all','math','code','general'],default='all')
    p.add_argument('--topup-tokens',type=int)
    p.add_argument('--deadline',type=float,required=True);args=p.parse_args()
    if args.domain!='all':
        acquire(args.domain,args.deadline,args.topup_tokens);return
    if args.topup_tokens is not None:raise ValueError('Choose one explicit domain for top-up')
    children=[]
    for domain in ['math','code','general']:
        log=(JOB/f'cpt-{domain}.log').open('a')
        process=subprocess.Popen([sys.executable,__file__,'--domain',domain,'--deadline',str(args.deadline)],stdout=log,stderr=subprocess.STDOUT)
        children.append((domain,process,log))
    failures=[]
    for domain,process,log in children:
        code=process.wait();log.close()
        if code:failures.append(dict(domain=domain,exit_code=code))
    write(JOB/'cpt-acquisition.json',dict(status='failed' if failures else 'complete',failures=failures,ready_for_training=False))
    if failures:raise RuntimeError('CPT source acquisition failed; preserve receipts')


if __name__=='__main__':main()
