"""CPU index of retained prior code/general responses; original data untouched."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
from screening import JOB,WORK,code_family,code_function_families
sys.path.insert(0,'/workspace/video-model')
from common import tokenizer
from evaluate import extract_code


def main():
    p=argparse.ArgumentParser();p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--function-level',action='store_true');args=p.parse_args()
    stem='prior-code-functions' if args.function_level else 'prior-code-families'
    db=sqlite3.connect(WORK/(stem+'.sqlite'));db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS families(hash TEXT PRIMARY KEY)')
    db.execute('CREATE TABLE IF NOT EXISTS cursors(source TEXT PRIMARY KEY,n INTEGER)')
    tok=tokenizer();counts=collections.Counter();sources=[]
    for source in ['code','r1-code','general']:
        folder=Path('/dev/shm/l20-posttrain/prepared')/source
        off=np.load(folder/'offsets.npy');ids=np.memmap(folder/'ids.u16',mode='r',dtype=np.uint16)
        masks=np.memmap(folder/'mask.u8',mode='r',dtype=np.uint8)
        cursor=db.execute('SELECT n FROM cursors WHERE source=?',(source,)).fetchone();start=cursor[0] if cursor else 0
        for begin in range(start,len(off)-1,256):
            if time.time()>args.deadline:raise TimeoutError('Prior code-family build bound reached')
            end=min(len(off)-1,begin+256);batch=[]
            for i in range(begin,end):
                a,b=int(off[i]),int(off[i+1]);batch.append(ids[a:b][masks[a:b]!=0].tolist())
            texts=tok.batch_decode(batch,skip_special_tokens=True)
            for text in texts:
                try:
                    code=extract_code(text)
                    keys=code_function_families(code) if args.function_level else {code_family(code)}
                    db.executemany('INSERT OR IGNORE INTO families VALUES(?)',[(key,) for key in keys]);counts['parsed']+=1
                except Exception:counts['unparseable_or_not_code']+=1
            db.execute('INSERT OR REPLACE INTO cursors VALUES(?,?)',(source,end));db.commit()
            if begin%8192==0:
                (JOB/(stem+'-status.json')).write_text(json.dumps(dict(source=source,indexed=end,source_total=len(off)-1,counts=dict(counts))))
        sources.append(dict(source=source,records=len(off)-1))
    count=db.execute('SELECT count(*) FROM families').fetchone()[0]
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
    path=WORK/(stem+'.sqlite')
    with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    (JOB/(stem+'-complete.json')).write_text(json.dumps(dict(status='complete',families=count,
       sources=sources,counts_this_attempt=dict(counts),sha256=sha,path=str(path),
       function_level=args.function_level,
       limitation='Canonical AST grouping covers retained code/general source responses, additionally embedded functions/classes when selected; semantic equivalence across different implementations cannot be exhaustively proved.'),indent=2))


if __name__=='__main__':main()
