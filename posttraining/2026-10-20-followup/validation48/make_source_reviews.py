"""Freeze fixed 200-row review samples only from completed immutable corpora.

Writing a sample never approves the source. Full review findings and independent
controls must be separately recorded before any quality gate can pass.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np

JOB = Path('/workspace/video-model/validation-48h')
WORK = Path('/dev/shm/video-model-validation48')
sys.path.insert(0, '/workspace/video-model')


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source', required=True,
       choices=['sft-math-replacement','sft-math-olympiads-v3','sft-math-olympiads-v4','sft-openr1-short','sft-math-calculus-cas-v1','sft-math-computational-v2','sft-code','sft-general','cpt-math','cpt-code','cpt-general'])
    args=p.parse_args(); source=args.source
    output=JOB/'reviews'/f'{source}-sample-200.jsonl';output.parent.mkdir(exist_ok=True)
    manifest=output.with_suffix('.manifest.json')
    if output.exists() or manifest.exists():
        raise RuntimeError('Existing review sample is frozen; inspect, do not resample')
    cpt=source.startswith('cpt-');domain=source.split('-',1)[1]
    if cpt:
        complete=JOB/f'cpt-{domain}-complete.json'
        data=WORK/'cpt-candidates'/domain/'documents.jsonl'
        identity=json.loads(complete.read_text());expected=identity['files']['documents.jsonl']['sha256']
    else:
        name=(domain+'-complete.json') if domain in ['math-replacement','math-olympiads-v3','math-olympiads-v4','openr1-short','math-calculus-cas-v1','math-computational-v2'] else f'candidates-{domain}-complete.json'
        complete=JOB/name
        identity=json.loads(complete.read_text());data=Path(identity['path']);expected=identity['sha256']
    if sha(data)!=expected:raise ValueError('Completed source content mismatch')
    rows=[json.loads(line) for line in data.read_text().splitlines()]
    if len(rows)<200:raise ValueError('Insufficient source rows for declared review')
    chosen=random.Random(20261023).sample(range(len(rows)),200)
    if cpt:
        from common import tokenizer
        tok=tokenizer();ids=np.memmap(data.parent/'ids.u16',dtype=np.uint16,mode='r')
    with output.open('w') as stream:
        for j,i in enumerate(chosen):
            row=dict(rows[i],review_index=j,source_line=i)
            if cpt:
                row['training_text']=tok.decode(ids[row['offset']:row['offset']+row['tokens']-1].tolist(),skip_special_tokens=False)
                row['view_semantics']='Exact decoded candidate training token sequence; original raw bytes are bound separately by source content hashes.'
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest.write_text(json.dumps(dict(source=source,seed=20261023,n=200,population=len(rows),
        source_sha256=expected,source_completion=str(complete),sample_sha256=sha(output),
        review_complete=False,quality_audit_passed=False),indent=2)+'\n')
    print(manifest.read_text())


if __name__=='__main__':main()
