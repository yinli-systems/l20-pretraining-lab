"""On-policy rejection-sampling fine-tuning with answer/execution rewards."""
import argparse, concurrent.futures, gc, json, multiprocessing, random, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from common import ROOT,CACHE,SEED,tokenizer,setup_torch,load_model,encode_messages,write_json,log
from evaluate import math_correct,code_correct,extract_code,generate_rows

def grade_math(item):
    row,text=item
    return math_correct(text,row['answer'])

def make_dataset(accepted,roundid,directory=None):
    tok=tokenizer(); rng=random.Random(SEED+roundid)
    dest=Path(directory) if directory is not None else CACHE/f'rft-{roundid}'; dest.mkdir(exist_ok=True); output={}
    for split in ['train','valid']:
        selected=[r for i,r in enumerate(accepted) if (i%20==0)==(split=='valid')]
        sequences=[]
        for r in selected:
            for a in [r['completion'],r['reference']]:
                ids,labels=encode_messages(tok,[{'role':'user','content':r['prompt']},{'role':'assistant','content':a}])
                if 32<=len(ids)<=2048: sequences.append((ids,[int(y!=-100) for y in labels]))
        rng.shuffle(sequences); blocks=[]; current=([],[])
        for ids,mask in sequences:
            if len(current[0])+len(ids)>2048:
                blocks.append(current); current=([],[])
            current[0].extend(ids); current[1].extend(mask)
        if current[0]: blocks.append(current)
        f=open(dest/f'{split}.ids.u16','wb'); fm=open(dest/f'{split}.mask.u8','wb')
        for ids,mask in blocks:
            n=2048-len(ids); np.asarray(ids+[tok.pad_token_id]*n,dtype=np.uint16).tofile(f); np.asarray(mask+[0]*n,dtype=np.uint8).tofile(fm)
        # Replay 30% of as many original packed blocks as RFT blocks.
        original_ids=np.memmap(CACHE/'packed'/f'{split}.ids.u16',mode='r',dtype=np.uint16).reshape(-1,2048)
        original_masks=np.memmap(CACHE/'packed'/f'{split}.mask.u8',mode='r',dtype=np.uint8).reshape(-1,2048)
        indices=rng.sample(range(len(original_ids)),min(len(original_ids),max(4,int(len(blocks)*0.3))))
        for i in indices: original_ids[i].tofile(f); original_masks[i].tofile(fm)
        f.close(); fm.close(); output[split]={'generated_prompts':len(selected),'blocks':len(blocks)+len(indices)}
    write_json(dest/'manifest.json',output); return dest,output

def main():
    p=argparse.ArgumentParser(); p.add_argument('--round',type=int,required=True); p.add_argument('--deadline',type=float,required=True); p.add_argument('--max-prompts',type=int,default=12000); a=p.parse_args()
    setup_torch(); rng=random.Random(SEED+a.round); tok=tokenizer(); pool=[]
    for source in ['math','code']:
        rows=[json.loads(x) for x in (CACHE/'prepared'/source/'pool.jsonl').read_text().splitlines()]
        rng.shuffle(rows); pool.extend(rows[:a.max_prompts//2])
    rng.shuffle(pool)
    if not (ROOT/'inference-profile.json').exists():
        try:
            with open(ROOT/'logs/inference-benchmark.log','w') as f:
                subprocess.run([sys.executable,str(ROOT/'inference_benchmark.py'),'--model',str(ROOT/'models/best')],stdout=f,stderr=subprocess.STDOUT,timeout=420,check=True)
        except Exception as e:
            write_json(ROOT/'inference-profile.json',{'selected':{'mode':'dynamic'},'fallback_reason':repr(e)})
    model=load_model(ROOT/'models/best',training=False,liger=False)
    accepted=[]; attempted=0; completions=0; correct=0
    out=ROOT/f'rft-{a.round}-accepted.jsonl'; allout=ROOT/f'rft-{a.round}-rollouts.jsonl'
    mathpool=concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'))
    codepool=concurrent.futures.ThreadPoolExecutor(max_workers=8)
    with open(out,'w') as acceptedfile,open(allout,'w') as allfile:
        for start in range(0,len(pool),8):
            if time.time()>a.deadline-90: break
            original=pool[start:start+8]; rows=[]
            for r in original:
                for _ in range(4): rows.append({'task':'humaneval' if r['source']=='code' else 'gsm8k','prompt':r['prompt'],'answer':r['answer']})
            texts=generate_rows(model,tok,rows,batch=32,max_new=1024,sample=True)
            pending=[]
            for i,(r,text) in enumerate(zip([x for x in original for _ in range(4)],texts)):
                if r['source']=='math': future=mathpool.submit(grade_math,(r,text))
                else:
                    tests=r['tests']; tests=json.loads(tests) if isinstance(tests,str) else tests
                    future=codepool.submit(code_correct,extract_code(text),'\n'.join(tests),'')
                pending.append(future)
            grades=[f.result(timeout=20) for f in pending]
            for j,r in enumerate(original):
                successes=[]
                for k in range(4):
                    i=j*4+k; ok=grades[i]; text=texts[i]; completions+=1; correct+=int(ok)
                    allfile.write(json.dumps({'prompt_hash':r['prompt_hash'],'source':r['source'],'completion':text,'correct':ok})+'\n')
                    if ok and len(tok.encode(text,add_special_tokens=False))>=64: successes.append(text)
                if successes:
                    chosen={**r,'completion':rng.choice(successes)}
                    ids,_=encode_messages(tok,[{'role':'user','content':r['prompt']},{'role':'assistant','content':chosen['completion']}])
                    if len(ids)<=2048: accepted.append(chosen); acceptedfile.write(json.dumps(chosen)+'\n'); acceptedfile.flush()
            attempted+=len(original)
            stats={'round':a.round,'attempted_prompts':attempted,'completions':completions,'correct_completions':correct,'accepted_prompts':len(accepted),'sample_accuracy':correct/max(1,completions)}
            write_json(ROOT/f'rft-{a.round}-progress.json',stats); log('verified_rft_sampling',**stats)
    mathpool.shutdown(); codepool.shutdown(); del model; gc.collect(); torch.cuda.empty_cache()
    if len(accepted)>=256:
        dest,manifest=make_dataset(accepted,a.round)
        write_json(ROOT/f'rft-{a.round}-ready.json',{'data':str(dest),'manifest':manifest,'accepted':len(accepted)})
    else: log('rft_skipped',reason='fewer than 256 verified complete responses',accepted=len(accepted))

if __name__=='__main__': main()
