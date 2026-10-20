"""Short, complete verified reasoning traces; never truncate a reasoning answer."""
import ast,collections,concurrent.futures,hashlib,json,os,random,shutil,time
from pathlib import Path
import numpy as np
from common import ROOT,CACHE,SEED,tokenizer,encode_messages,write_json,log
from prepare import Screen,digest

def build(source,repo,config,split,target,maxscan):
    dest=CACHE/'prepared'/source
    if (dest/'complete').exists(): log('advanced_source_reused',source=source); return
    from datasets import load_dataset
    from huggingface_hub import HfApi
    tok=tokenizer(); screen=Screen([json.loads(x) for x in (ROOT/'benchmarks.jsonl').read_text().splitlines()])
    prior=ROOT/f'data-{source}.json'
    revision=json.loads(prior.read_text())['revision'] if prior.exists() else HfApi().dataset_info(repo).sha
    ds=load_dataset(repo,config,split=split,streaming=True,revision=revision).shuffle(seed=SEED,buffer_size=1024)
    dest.mkdir(parents=True,exist_ok=True)
    f=open(dest/'ids.u16','wb'); fm=open(dest/'mask.u8','wb'); offsets=[0]; validation=[]; seen=set(); counts=collections.Counter()
    for i,row in enumerate(ds):
        if counts['scanned']>=maxscan: break
        counts['scanned']+=1
        if counts['scanned']%10000==0: log('advanced_scan',source=source,**counts)
        if source=='r1-math':
            p=row['problem']; good=[]
            complete=row.get('is_reasoning_complete',[]); correct=row.get('correctness_math_verify',[])
            for j,text in enumerate(row.get('generations',[])):
                if j<len(correct) and correct[j] and j<len(complete) and complete[j] and 200<len(text)<11000 and '\\boxed' in text: good.append(text)
            if not good: continue
            a=min(good,key=len)
        else:
            if row.get('split')!='train': continue
            p=row.get('input',''); a=row.get('output','')
            if not p or p=='-' or not row.get('solution') or not (200<len(a)<11000): continue
            if '```' not in a or ('<think>' in a and '</think>' not in a): continue
        h=digest(p)
        if h in seen: counts['duplicate_prompt']+=1; continue
        if screen.matches(p): counts['benchmark_rejected']+=1; continue
        if len(p)>6500: counts['prompt_length_rejected']+=1; continue
        ids,labels=encode_messages(tok,[{'role':'user','content':p},{'role':'assistant','content':a}])
        if len(ids)>2048: counts['length_rejected']+=1; continue
        seen.add(h); np.asarray(ids,dtype=np.uint16).tofile(f); np.asarray([x!=-100 for x in labels],dtype=np.uint8).tofile(fm)
        offsets.append(offsets[-1]+len(ids)); validation.append(int(h[:8],16)%1000<3)
        counts['accepted']+=1; counts['tokens']+=len(ids); counts['supervised_tokens']+=sum(x!=-100 for x in labels)
        if counts['scanned']%1000==0: log('advanced_progress',source=source,**counts)
        if counts['accepted']>=target or counts['scanned']>=maxscan: break
    f.close(); fm.close(); np.save(dest/'offsets.npy',np.asarray(offsets,dtype=np.int64)); np.save(dest/'validation.npy',np.asarray(validation,dtype=np.uint8))
    write_json(ROOT/f'data-{source}.json',{'repo':repo,'revision':revision,'config':config,'split':split,'counts':dict(counts),'max_length':2048,'filter':'complete correct R1 math traces (math-verify flag), or released R1 code solutions from source train split; benchmark screening, per-source normalized prompt dedup, no truncation'})
    (dest/'complete').touch(); log('advanced_complete',source=source,**counts)

def repack():
    from prepare import SOURCES
    from evaluate import extract_code
    tok=tokenizer(); rng=np.random.default_rng(SEED); sources={}; train=[]; valid=[]; seen={}; duplicates=0; syntax_rejected=0
    # Keep the complete R1 trace when a source repeats an existing short answer.
    for source in ['r1-math','r1-code',*SOURCES]:
        dest=CACHE/'prepared'/source
        ids=np.memmap(dest/'ids.u16',mode='r',dtype=np.uint16); masks=np.memmap(dest/'mask.u8',mode='r',dtype=np.uint8)
        offsets=np.load(dest/'offsets.npy'); flags=np.load(dest/'validation.npy'); sources[source]=(ids,masks,offsets)
        for i,v in enumerate(flags):
            a,b=int(offsets[i]),int(offsets[i+1])
            if source=='r1-code':
                text=tok.decode(ids[a:b][masks[a:b]==1].tolist(),skip_special_tokens=True)
                try: ast.parse(extract_code(text))
                except (SyntaxError,ValueError): syntax_rejected+=1; continue
            h=hashlib.sha256(ids[a:b][masks[a:b]==0].tobytes()).digest()
            if h in seen:
                if seen[h]!=bool(v): raise RuntimeError('Cross-source validation grouping mismatch')
                duplicates+=1; continue
            seen[h]=bool(v); (valid if v else train).append((source,i))
    dest=CACHE/'packed-advanced'; dest.mkdir(exist_ok=True); tok=tokenizer(); results={}
    for split,rows in [('train',train),('valid',valid)]:
        rng.shuffle(rows); f=open(dest/f'{split}.ids.u16','wb'); fm=open(dest/f'{split}.mask.u8','wb'); bins=[]; n=0; ntokens=0; supervised=0
        def emit(block):
            nonlocal n,ntokens,supervised
            x,m=block; length=len(x); n+=1; ntokens+=length; supervised+=sum(m)
            np.asarray(x+[tok.pad_token_id]*(2048-length),dtype=np.uint16).tofile(f); np.asarray(m+[0]*(2048-length),dtype=np.uint8).tofile(fm)
        for source,i in rows:
            x,m,off=sources[source]; a,b=int(off[i]),int(off[i+1]); length=b-a
            possible=[(2048-len(z[0])-length,j) for j,z in enumerate(bins) if len(z[0])+length<=2048]
            if possible: j=min(possible)[1]
            else:
                if len(bins)>=128: emit(bins.pop(max(range(len(bins)),key=lambda j:len(bins[j][0]))))
                bins.append(([],[])); j=len(bins)-1
            bins[j][0].extend(x[a:b].tolist()); bins[j][1].extend(m[a:b].tolist())
            if len(bins[j][0])==2048: emit(bins.pop(j))
        for block in bins: emit(block)
        f.close(); fm.close(); results[split]={'examples':len(rows),'blocks':n,'tokens':ntokens,'supervised_tokens':supervised,'packing_efficiency':ntokens/(n*2048)}
    write_json(dest/'manifest.json',results)
    # All LR pilots must use exactly the original dataset before main SFT switches.
    while not (ROOT/'pilot-0.0002-complete.json').exists(): time.sleep(5)
    old=CACHE/'packed-original'
    if old.exists(): raise RuntimeError('Packed data switch already performed')
    (CACHE/'packed').rename(old); dest.rename(CACHE/'packed')
    manifest={'sequence_length':2048,'seed':SEED,'splits':results,'global_token_prompt_duplicates_removed':duplicates,'r1_code_syntax_rejected':syntax_rejected,'packing':'complete ChatML conversations with standard causal concatenation and assistant-only loss; no answer truncation; prompt-grouped holdout','advanced_sources':['r1-math','r1-code'],'verification':'R1 math uses completion and math-verify flags; R1 code uses complete released traces and Python syntax checks, not independent execution verification; execution verification is used for online code RFT'}
    write_json(ROOT/'data-manifest.json',manifest); write_json(ROOT/'advanced-data-complete.json',manifest)
    (CACHE/'packed'/'advanced-complete').touch(); (CACHE/'packed'/'complete').touch(); log('advanced_data_activated',**manifest)

def main():
    try:
        # Resolve Transformers' lazy exports on the main thread before workers.
        from datasets import load_dataset
        from transformers import AutoTokenizer
        from huggingface_hub import HfApi
        tokenizer()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            jobs=[ex.submit(build,'r1-math','open-r1/OpenR1-Math-220k','default','train',15000,95000),ex.submit(build,'r1-code','nvidia/OpenCodeReasoning','split_0','split_0',10000,200000)]
            for f in jobs: f.result()
        repack()
    except Exception as e:
        write_json(ROOT/'advanced-data-error.json',{'error':repr(e),'fallback':'original audited packed data retained'})
        # Optional enrichment must not block main training indefinitely.
        (CACHE/'packed'/'advanced-complete').touch(); raise

if __name__=='__main__': main()
