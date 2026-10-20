"""Pinned, prompt-grouped, benchmark-screened reasoning data preparation."""
import argparse, collections, hashlib, json, random, re, time
from pathlib import Path
import numpy as np
from common import ROOT, CACHE, SEED, tokenizer, write_json, log

def norm(s): return ' '.join(re.findall(r'\w+', s.lower()))
def digest(s): return hashlib.sha256(norm(s).encode()).hexdigest()
def grams(s, n=13):
    w=norm(s).split(); return {' '.join(w[i:i+n]) for i in range(max(0,len(w)-n+1))}

def benchmarks():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api=HfApi(); records=[]; revisions={}
    pinned=json.loads((ROOT/'benchmark-provenance.json').read_text()) if (ROOT/'benchmark-provenance.json').exists() else {}
    specs=[('gsm8k','openai/gsm8k','main','test'),('math500','HuggingFaceH4/MATH-500',None,'test'),('humaneval','openai/openai_humaneval',None,'test')]
    for task,repo,config,split in specs:
        rev=pinned.get(repo) or api.dataset_info(repo).sha; revisions[repo]=rev
        ds=load_dataset(repo,config,split=split,revision=rev)
        rows=list(ds); random.Random(SEED).shuffle(rows)
        ndev={'gsm8k':96,'math500':64,'humaneval':32}[task]
        for i,r in enumerate(rows):
            p=r.get('question',r.get('problem',r.get('prompt','')))
            if task=='gsm8k': ans=r['answer'].split('####')[-1].strip()
            else: ans=r.get('answer','')
            records.append({'task':task,'split':'dev' if i<ndev else 'final','prompt':p,'answer':ans, 'id':r.get('task_id',digest(p)), 'test':r.get('test',''),'entry_point':r.get('entry_point','')})
    out=ROOT/'benchmarks.jsonl'
    out.write_text(''.join(json.dumps(r)+'\n' for r in records))
    write_json(ROOT/'benchmark-provenance.json',revisions)
    return records

class Screen:
    def __init__(self, rows):
        self.hashes={digest(r['prompt']) for r in rows}
        self.ng=set().union(*(grams(r['prompt']) for r in rows))
        self.short={norm(r['prompt']) for r in rows if 12<=len(norm(r['prompt']))<80}
    def matches(self,p):
        t=norm(p)
        return digest(p) in self.hashes or bool(grams(p)&self.ng) or any(s==t or s in t for s in self.short)

SOURCES={
 'math':('nvidia/OpenMathInstruct-2',None,'train_1M',400000,1200000),
 'code':('nvidia/OpenCodeInstruct',None,'train',300000,1500000),
 'numina':('AI-MO/NuminaMath-CoT',None,'train',70000,400000),
 'general':('HuggingFaceTB/smoltalk','smol-magpie-ultra','train',20000,45000),
}

def convert(source,r):
    if source=='math':
        p,a=r['problem'],r['generated_solution']
        if not r.get('expected_answer') or '\\boxed' not in a: return None
        msgs=[{'role':'user','content':p},{'role':'assistant','content':a}]
    elif source=='code':
        try:
            if float(r.get('average_test_score',0))<1: return None
            tests=json.loads(r.get('unit_tests','[]')) if isinstance(r.get('unit_tests'),str) else r.get('unit_tests',[])
            status=json.loads(r.get('tests_execution_status','[]')) if isinstance(r.get('tests_execution_status'),str) else r.get('tests_execution_status',[])
            if not tests or len(tests)<3 or any(x!='pass' for x in status): return None
            judge=json.loads(r.get('llm_judgement','{}')) if isinstance(r.get('llm_judgement'),str) else r.get('llm_judgement',{})
            if judge and judge.get('logical_correctness',{}).get('score',0)<4: return None
        except (ValueError,TypeError): return None
        p,a=r['input'],r['output']; msgs=[{'role':'user','content':p},{'role':'assistant','content':a}]
    else:
        msgs=r.get('messages',[])
        if not msgs or msgs[-1]['role']!='assistant': return None
        p='\n'.join(m['content'] for m in msgs if m['role']=='user')
        a=msgs[-1]['content']
        if source=='numina' and ('\\boxed' not in a or r.get('source') in ['olympiads','amc_aime']): return None
    if not p.strip() or not a.strip() or len(p)>7500 or len(a)<40 or len(a)>13000: return None
    return p,msgs

def source_main(source):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    tok=tokenizer(); screen=Screen([json.loads(x) for x in (ROOT/'benchmarks.jsonl').read_text().splitlines()])
    repo,cfg,split,target,maxscan=SOURCES[source]
    prior=ROOT/f'data-{source}.json'
    rev=json.loads(prior.read_text())['revision'] if prior.exists() else HfApi().dataset_info(repo).sha
    ds=load_dataset(repo,cfg,split=split,streaming=True,revision=rev).shuffle(seed=SEED,buffer_size=8192)
    dest=CACHE/'prepared'/source; dest.mkdir(parents=True,exist_ok=True)
    seen=set(); counts=collections.Counter(); batch=[]; lens=[]; groups=[]; offsets=[0]; pool=[]
    bos=tok.bos_token or ''; imstart=tok.convert_tokens_to_ids('<|im_start|>'); imend=tok.eos_token_id
    ah=tok.encode('<|im_start|>assistant\n',add_special_tokens=False)
    idsfile=open(dest/'ids.u16','wb'); maskfile=open(dest/'mask.u8','wb')
    def flush():
        texts=[tok.apply_chat_template(m,tokenize=False,add_generation_prompt=False) for p,m,r in batch]
        encoded=tok(texts,add_special_tokens=False,return_attention_mask=False)['input_ids']
        for (p,msgs,r),ids in zip(batch,encoded):
            if len(ids)>2048 or len(ids)<32: counts['length_rejected']+=1; continue
            mask=np.zeros(len(ids),dtype=np.uint8)
            for j,x in enumerate(ids):
                if x==imstart and ids[j:j+len(ah)]==ah:
                    k=j+len(ah)
                    while k<len(ids):
                        mask[k]=1
                        if ids[k]==imend: break
                        k+=1
            if mask.sum()<20: counts['empty_rejected']+=1; continue
            np.asarray(ids,dtype=np.uint16).tofile(idsfile); mask.tofile(maskfile)
            offsets.append(offsets[-1]+len(ids)); lens.append(len(ids))
            h=digest(p); isval=int(h[:8],16)%1000<3
            groups.append(isval); counts['accepted']+=1; counts['tokens']+=len(ids); counts['supervised_tokens']+=int(mask.sum())
            if isval: counts['validation_examples']+=1
            if not isval and len(pool)<16000 and source in ['math','code']:
                pool.append({'source':source,'prompt':p,'answer':r.get('expected_answer',''),'tests':r.get('unit_tests',[]),'reference':msgs[-1]['content'],'prompt_hash':h})
        batch.clear()
    for i,r in enumerate(ds):
        counts['scanned']+=1
        row=convert(source,r)
        if row:
            p,msgs=row; h=digest(p)
            if h in seen: counts['duplicate_prompt']+=1
            elif screen.matches(p): counts['benchmark_rejected']+=1
            else:
                seen.add(h); batch.append((p,msgs,r))
        if len(batch)>=256: flush()
        if i%10000==0: log('source_progress',source=source,**counts)
        if counts['accepted']>=target or counts['scanned']>=maxscan: break
    if batch: flush()
    idsfile.close(); maskfile.close()
    np.save(dest/'offsets.npy',np.asarray(offsets,dtype=np.int64)); np.save(dest/'validation.npy',np.asarray(groups,dtype=np.uint8))
    (dest/'pool.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pool))
    manifest={'source':source,'repo':repo,'revision':rev,'config':cfg,'split':split,'seed':SEED,'max_length':2048,'counts':dict(counts),'filter':'prompt exact dedup; exact/13-word benchmark screening; no truncation; code supplied tests all pass and judge >=4'}
    write_json(ROOT/f'data-{source}.json',manifest)
    (dest/'complete').touch(); log('source_complete',**manifest)

def pack():
    rng=np.random.default_rng(SEED); sources={}; train=[]; valid=[]
    for s in SOURCES:
        p=CACHE/'prepared'/s
        if not (p/'complete').exists(): raise RuntimeError(f'{s} not ready')
        off=np.load(p/'offsets.npy'); val=np.load(p/'validation.npy')
        sources[s]=(np.memmap(p/'ids.u16',mode='r',dtype=np.uint16),np.memmap(p/'mask.u8',mode='r',dtype=np.uint8),off)
        for i in range(len(val)): (valid if val[i] else train).append((s,i))
    tok=tokenizer(); result={}
    for split,rows in [('train',train),('valid',valid)]:
        rng.shuffle(rows); dest=CACHE/'packed'; dest.mkdir(exist_ok=True)
        fids=open(dest/f'{split}.ids.u16','wb'); fm=open(dest/f'{split}.mask.u8','wb')
        bins=[]; nblocks=0; ntokens=0; supervised=0
        def emit(b):
            nonlocal nblocks,ntokens,supervised
            ids,mask=b; n=len(ids); ntokens+=n; supervised+=sum(mask); nblocks+=1
            np.asarray(ids+[tok.pad_token_id]*(2048-n),dtype=np.uint16).tofile(fids)
            np.asarray(mask+[0]*(2048-n),dtype=np.uint8).tofile(fm)
        for s,i in rows:
            ids,m,off=sources[s]; a,b=int(off[i]),int(off[i+1]); length=b-a
            fits=[(2048-len(v[0])-length,j) for j,v in enumerate(bins) if len(v[0])+length<=2048]
            if fits: j=min(fits)[1]
            else:
                if len(bins)>=128: emit(bins.pop(max(range(len(bins)),key=lambda j:len(bins[j][0]))))
                bins.append(([],[])); j=len(bins)-1
            bins[j][0].extend(ids[a:b].tolist()); bins[j][1].extend(m[a:b].tolist())
            if len(bins[j][0])==2048: emit(bins.pop(j))
        for block in bins: emit(block)
        fids.close(); fm.close()
        result[split]={'examples':len(rows),'blocks':nblocks,'tokens':ntokens,'supervised_tokens':supervised,'packing_efficiency':ntokens/(nblocks*2048)}
    write_json(ROOT/'data-manifest.json',{'sequence_length':2048,'seed':SEED,'splits':result,'packing':'complete conversations, ChatML EOS separators, standard causal concatenation, no answer truncation; prompts masked from loss; prompt-grouped holdout'})
    write_json(CACHE/'packed/manifest.json',result)
    (CACHE/'packed/complete').touch(); log('packing_complete',**result)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['benchmarks','pack',*SOURCES]); a=p.parse_args()
    if a.mode=='benchmarks': benchmarks()
    elif a.mode=='pack': pack()
    else: source_main(a.mode)
