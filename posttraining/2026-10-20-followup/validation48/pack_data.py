"""Stage exact native-2048 objectives after conservative overlap screening.

Staged manifests always have quality_audit_passed=false. This utility cannot
approve its own inputs; source/gold audits and an immutable suite freeze remain
separate prerequisites. No failed/blocklisted candidate can be read as training.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import sys
import time
import numpy as np

JOB=Path('/workspace/video-model/validation-48h')
WORK=Path('/dev/shm/video-model-validation48')
sys.path.insert(0,'/workspace/video-model')
from screening import norm,digest,family,code_family,code_function_families
from build_ngram_index import spans
from cpt_inputs import checked_segments
from cpt_content_checks import content_issue
from common import tokenizer,encode_messages


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


class Screen:
    def __init__(self,suites):
        self.old=np.load(WORK/'prior-13word-hashes.npy',mmap_mode='r')
        self.db=sqlite3.connect('file:'+str(WORK/'prior-prompts.sqlite')+'?mode=ro',uri=True)
        self.astdb=sqlite3.connect('file:'+str(WORK/'prior-code-families.sqlite')+'?mode=ro',uri=True)
        if not (JOB/'prior-code-functions-complete.json').exists():
            raise RuntimeError('Embedded-function exclusion index is not complete')
        self.functiondb=sqlite3.connect('file:'+str(WORK/'prior-code-functions.sqlite')+'?mode=ro',uri=True)
        self.new=set();self.families=set();self.code_families=set();self.code_functions=set()
        for path in suites:
            for line in Path(path).read_text().splitlines():
                row=json.loads(line)
                text=row.get('source_prompt',row.get('prompt',row.get('question','')))
                if 'options' in row:text+=' '+' '.join(row['options'])
                self.new.update(spans(norm(text)));self.families.add(family(text))
                if row.get('domain')=='python_functions':
                    self.code_families.add(row['family'])
                    self.code_functions.update(code_function_families(row['reference']))
    def overlap(self,text,old_prompt=True):
        if family(text) in self.families:return True
        if old_prompt and self.db.execute('SELECT 1 FROM prompts WHERE hash=? OR family_hash=? LIMIT 1',(digest(text),family(text))).fetchone():return True
        values=np.asarray(spans(norm(text)),dtype=np.uint64)
        if any(int(x) in self.new for x in values):return True
        positions=np.searchsorted(self.old,values)
        valid=positions<len(self.old)
        return bool((self.old[positions[valid]]==values[valid]).any())
    def code_overlap(self,reference):
        f=code_family(reference)
        embedded=code_function_families(reference)
        # Every canonical family was allocated before any model score. Code
        # holdouts cannot migrate into training through another prompt.
        return f,(int(f[:12],16)%10000<3000 or f in self.code_families or
                  bool(self.astdb.execute('SELECT 1 FROM families WHERE hash=?',(f,)).fetchone()) or
                  self.function_overlap(embedded))
    def function_overlap(self,keys):
        return bool(keys & self.code_functions or any(
            self.functiondb.execute('SELECT 1 FROM families WHERE hash=?',(key,)).fetchone() for key in keys))


def checked_input(path):
    path=Path(path);identity=sha(path)
    blocked=json.loads((JOB/'training-input-blocklist.json').read_text())['blocked']
    if any(str(path.resolve())==r['path'] or identity==r['sha256'] for r in blocked):
        raise ValueError('Quarantined failed source is prohibited as training input')
    return identity


def stage_sft(plan,screen,out,deadline):
    tok=tokenizer();counts=collections.Counter();pools={};identities={}
    for domain in ['math','code','general']:
        path=Path(plan['sft'][domain]);identities[str(path)]=checked_input(path)
        pool=[];seen=set();seen_code=set()
        with path.open() as stream:
            for line_number,line in enumerate(stream):
                if time.time()>deadline:raise TimeoutError('Bounded SFT preparation expired')
                row=json.loads(line);p=row['prompt'];counts[domain+':input']+=1
                if screen.overlap(p):counts[domain+':overlap']+=1;continue
                fam=family(p)
                if fam in seen:counts[domain+':duplicate_numeric_family']+=1;continue
                text=p+' '+row['response']
                if any(tag in text for tag in ['<|im_start|>','<|im_end|>','<|endoftext|>']):
                    counts[domain+':literal_chat_control']+=1;continue
                if len(re.findall(r'[\u3400-\u9fff]',text))>4:
                    counts[domain+':language']+=1;continue
                codefam=None
                if domain=='code':
                    try:codefam,overlap=screen.code_overlap(row['reference'])
                    except Exception:counts['code:AST_invalid']+=1;continue
                    if overlap or codefam in seen_code:counts['code:old_eval_or_duplicate_AST']+=1;continue
                messages=row.get('messages') or [dict(role='user',content=p),dict(role='assistant',content=row['response'])]
                # A general-instruction answer can itself quote a held-out
                # question. Check every retained turn, not just user prompts.
                if any(screen.overlap(m['content'],old_prompt=False) for m in messages):
                    counts[domain+':conversation_content_overlap']+=1;continue
                ids,labels=encode_messages(tok,messages);mask=[int(x!=-100) for x in labels];supervised=sum(mask)
                minimum=32 if domain=='math' else 200
                if len(ids)>2048 or not minimum<=supervised<=1000:
                    counts[domain+':length']+=1;continue
                seen.add(fam)
                if codefam:seen_code.add(codefam)
                record=dict(domain=domain,source_line=line_number,prompt_hash=digest(p),family=fam,
                    code_family=codefam,tokens=len(ids),supervised_tokens=supervised)
                pool.append((record,ids,mask));counts[domain+':eligible']+=1
        random.Random(20261020).shuffle(pool);pools[domain]=pool
    weights=dict(math=.4,code=.45,general=.15)
    available={d:sum(r[0]['supervised_tokens'] for r in rows) for d,rows in pools.items()}
    if not all(available.values()):raise ValueError('A required SFT domain has no eligible data')
    total=min(available[d]/weights[d] for d in weights);selected=[]
    for d,rows in pools.items():
        current=0;target=int(total*weights[d])
        for item in rows:
            n=item[0]['supervised_tokens']
            if current+n>target:continue
            selected.append(item);current+=n
        counts[d+':selected_supervised_tokens']=current
    random.Random(20261020).shuffle(selected)
    ids_file=(out/'train.ids.u16').open('wb');mask_file=(out/'train.mask.u8').open('wb')
    index=(out/'sample-index.jsonl').open('w');block=[];mask=[];blocks=0;nonpad=0;shifted=0
    def emit():
        nonlocal block,mask,blocks,nonpad,shifted
        if not block:return
        nonpad+=len(block);shifted+=sum(mask[1:])
        np.asarray(block+[0]*(2048-len(block)),dtype=np.uint16).tofile(ids_file)
        np.asarray(mask+[0]*(2048-len(mask)),dtype=np.uint8).tofile(mask_file)
        blocks+=1;block=[];mask=[]
    for record,ids,labels in selected:
        if len(block)+len(ids)>2048:emit()
        index.write(json.dumps(dict(record,block=blocks,start=len(block)))+'\n')
        block.extend(ids);mask.extend(labels)
    emit();ids_file.close();mask_file.close();index.close()
    return dict(objective='sft',context=2048,quality_audit_passed=False,input_sha256=identities,
         counts=dict(counts),available_supervised_tokens=available,unique_samples=len(selected),
         blocks=blocks,packed_tokens=blocks*2048,nonpadding_tokens=nonpad,
         supervised_tokens=sum(r[0]['supervised_tokens'] for r in selected),shifted_supervised_tokens=shifted,
         target_supervised_domain_mix=weights,within_epoch_repetition=False,
         complete_sample_bound=2048,packing='whole ChatML samples, pad explicit mask0, no sample truncation')


def stage_cpt(plan,screen,out,deadline):
    tok=tokenizer();streams={};counts=collections.Counter();identity={};seen=set();seen_modules=set()
    # Counts are exact packed-token limits, not unique-source-token claims.
    targets={'math':156250,'code':156250,'general':78125};total_blocks=sum(targets.values())
    temp=out/'domains';temp.mkdir()
    for domain in targets:
        segments=checked_segments(plan['cpt'][domain],domain,JOB)
        for _,_,identities in segments:identity.update(identities)
        buffer=[];nblocks=0
        dest=(temp/(domain+'.u16')).open('wb');index=(out/(domain+'-document-index.jsonl')).open('w')
        for segment_number,(root,_,_) in enumerate(segments):
            source=np.memmap(root/'ids.u16',dtype=np.uint16,mode='r')
            for line in (root/'documents.jsonl').open():
                if time.time()>deadline:raise TimeoutError('Bounded CPT screening expired')
                row=json.loads(line);counts[domain+':input_documents']+=1
                h='raw:'+row['normalized_text_sha256']
                if h in seen:counts[domain+':duplicate_document']+=1;continue
                tokens=source[row['offset']:row['offset']+row['tokens']].tolist()
                if len(tokens)!=row['tokens'] or not tokens or tokens[-1]!=0:
                    raise ValueError('CPT document offset/length or native EOS mismatch')
                text=tok.decode(tokens[:-1],skip_special_tokens=False)
                decoded='decoded:'+hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
                if decoded in seen:counts[domain+':duplicate_decoded_document']+=1;continue
                if any(tag in text for tag in ['<|im_start|>','<|im_end|>','<|endoftext|>']):
                    counts[domain+':literal_chat_control']+=1;continue
                issue=content_issue(text,domain)
                if issue:counts[domain+':'+issue]+=1;continue
                if screen.overlap(text,old_prompt=False):counts[domain+':prior_prompt_or_evaluation_span_overlap']+=1;continue
                # Same canonical Python implementation cannot enter through CPT.
                if domain=='code':
                    try:
                        keys=code_function_families(text)
                        if screen.function_overlap(keys):counts['code:prior_or_eval_embedded_AST_overlap']+=1;continue
                        module_family=code_family(text)
                        if module_family in seen_modules:counts['code:duplicate_module_AST_family']+=1;continue
                    except Exception:counts['code:decoded_invalid_AST']+=1;continue
                    seen_modules.add(module_family)
                seen.update([h,decoded]);counts[domain+':eligible_documents']+=1;counts[domain+':unique_eligible_tokens']+=len(tokens)
                index.write(json.dumps(dict(row,segment=segment_number,source_root=str(root)))+'\n');buffer.extend(tokens)
                whole=min(len(buffer)//2048,targets[domain]-nblocks)
                if whole:
                    np.asarray(buffer[:whole*2048],dtype=np.uint16).tofile(dest)
                    del buffer[:whole*2048];nblocks+=whole
                if nblocks==targets[domain]:break
            del source
            if nblocks==targets[domain]:break
        dest.close();index.close()
        if nblocks!=targets[domain]:raise ValueError(f'{domain} has only {nblocks} clean blocks; required {targets[domain]}')
        streams[domain]=np.memmap(temp/(domain+'.u16'),dtype=np.uint16,mode='r').reshape(-1,2048)
    order=[d for d,n in targets.items() for _ in range(n)];random.Random(20261020).shuffle(order)
    cursors=collections.Counter()
    with (out/'train.ids.u16').open('wb') as ids_file,(out/'train.mask.u8').open('wb') as mask_file:
        mask=np.ones(2048,dtype=np.uint8)
        for domain in order:
            streams[domain][cursors[domain]].tofile(ids_file);mask.tofile(mask_file);cursors[domain]+=1
    streams.clear()
    for path in temp.iterdir():path.unlink()
    temp.rmdir()
    return dict(objective='cpt',context=2048,quality_audit_passed=False,input_sha256=identity,
        counts=dict(counts),blocks=total_blocks,packed_tokens=total_blocks*2048,
        nonpadding_tokens=total_blocks*2048,supervised_tokens=total_blocks*2048,
        shifted_supervised_tokens=total_blocks*2047,domain_blocks=targets,
        notes=['CPT continuous chunks may cross document boundaries; native endoftext0 is supervised.',
               'Exact/normalized-content dedup and evaluation overlap screens do not exhaust semantic near-duplicates.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--objective',choices=['sft','cpt'],required=True)
    p.add_argument('--deadline',type=float,required=True);args=p.parse_args()
    plan_path=Path(args.plan);plan=json.loads(plan_path.read_text());out=WORK/'packed-provisional'/args.objective
    if out.exists():raise RuntimeError('Existing staged output; inspect before any replacement')
    out.mkdir(parents=True);screen=Screen(plan['suite_exclusion_inputs'])
    value=(stage_sft if args.objective=='sft' else stage_cpt)(plan,screen,out,args.deadline)
    value.update(plan_sha256=sha(plan_path),suite_exclusion_inputs={p:sha(p) for p in plan['suite_exclusion_inputs']},
         files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file()},
         remaining_gates=['independent source200 reviews','source/template near-duplicate review',
                          'reviewed immutable suite freeze','token/mask native CPU replay','storage and monitor capacity'])
    write(out/'manifest.json',value);print(json.dumps(value),flush=True)


if __name__=='__main__':main()
