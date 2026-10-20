import argparse, ast, concurrent.futures, gc, json, os, re, subprocess, sys, time
import multiprocessing
from pathlib import Path
import torch
from common import ROOT, CACHE, SEED, tokenizer, setup_torch, load_model, write_json, log

def math_correct(text,answer):
    try:
        from math_verify import parse, verify
        from math_verify import LatexExtractionConfig, ExprExtractionConfig
        gold=parse('\\boxed{'+str(answer)+'}',extraction_config=[LatexExtractionConfig()])
        pred=parse(text,extraction_config=[LatexExtractionConfig(),ExprExtractionConfig()],extraction_mode='first_match')
        return bool(gold and pred and verify(gold,pred,timeout_seconds=2))
    except Exception: return False

def extract_code(text,prompt='',base=False,entry_point=''):
    if base: return prompt+text
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,re.S)
    selected=None
    if blocks:
        if entry_point:
            for block in blocks:
                try: tree=ast.parse(block)
                except (SyntaxError,ValueError): continue
                if any(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==entry_point for node in tree.body):
                    selected=block; break
        if selected is None: selected=max(blocks,key=len)
    else:
        start=re.search(r'(?m)^(?:from \w+ import |import |def |class )',text)
        selected=text[start.start():] if start else text
    # Retain supplied imports/type declarations, just as the completion protocol does.
    return prompt+'\n'+selected if entry_point else selected

def code_correct(code,test,entry_point=''):
    payload={'code':code,'test':test,'entry_point':entry_point,'jail':str(CACHE/'empty-jail')}
    try:
        r=subprocess.run([sys.executable,'-I',str(ROOT/'sandbox.py')],input=json.dumps(payload),text=True,capture_output=True,timeout=6,env={'PATH':'/usr/bin:/bin','OPENBLAS_NUM_THREADS':'1'})
        return r.returncode==0 and json.loads(r.stdout.strip().splitlines()[-1]).get('passed',False)
    except Exception: return False

def grade_math_pair(pair):
    row,text=pair
    return {**row,'completion':text,'correct':math_correct(text,row['answer'])}

def prompt_text(tok,row,base=False):
    if base:
        if row['task']=='humaneval': return row['prompt']
        return 'Question: '+row['prompt']+'\nAnswer: Let\'s think step by step.\n'
    if row['task']=='humaneval': p='Complete the following Python function. Explain the approach briefly, then provide the complete Python code in a code block.\n\n'+row['prompt']
    else: p=row['prompt']+'\n\nSolve step by step and put your final answer in \\boxed{}.'
    return tok.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True)

def generate_rows(model,tok,rows,base=False,batch=32,max_new=1800,sample=False):
    model.eval(); tok.padding_side='left'; tok.truncation_side='left'
    profile=ROOT/'inference-profile.json'
    use_static=not base and profile.exists() and json.loads(profile.read_text()).get('selected',{}).get('mode')=='static'
    cache=None
    if use_static:
        from transformers import StaticCache,CompileConfig
        cache=StaticCache(config=model.config,max_cache_len=2048)
    order=sorted(range(len(rows)),key=lambda i:len(rows[i]['prompt']))
    outputs=['']*len(rows)
    for begin in range(0,len(order),batch):
        idx=order[begin:begin+batch]; texts=[prompt_text(tok,rows[i],base) for i in idx]
        inputs=tok(texts,return_tensors='pt',padding=True,add_special_tokens=False,truncation=True,max_length=1100).to('cuda')
        inputs.pop('token_type_ids',None)
        new=min(max_new,2048-inputs.input_ids.shape[1])
        extra={}
        if cache is not None and len(idx)==batch:
            cache.reset(); extra={'past_key_values':cache,'compile_config':CompileConfig(dynamic=False,mode='reduce-overhead')}
        with torch.inference_mode():
            kwargs=dict(max_new_tokens=new,do_sample=sample,temperature=0.8 if sample else None,top_p=0.95 if sample else None,use_cache=True,eos_token_id=[tok.eos_token_id,0],pad_token_id=tok.pad_token_id)
            try: result=model.generate(**inputs,**kwargs,**extra)
            except Exception as e:
                if not extra: raise
                log('static_cache_fallback',error=repr(e)); cache=None; torch.cuda.empty_cache()
                write_json(ROOT/'inference-profile.json',{'selected':{'mode':'dynamic'},'runtime_fallback_reason':repr(e)})
                result=model.generate(**inputs,**kwargs)
        decoded=tok.batch_decode(result[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
        for i,text in zip(idx,decoded): outputs[i]=text
        log('generation_progress',completed=min(begin+batch,len(rows)),total=len(rows))
    return outputs

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model'); p.add_argument('--base',action='store_true'); p.add_argument('--split',default='dev'); p.add_argument('--output',required=True); p.add_argument('--limit',type=int); a=p.parse_args()
    setup_torch(); tok=tokenizer(); rows=[json.loads(s) for s in (ROOT/'benchmarks.jsonl').read_text().splitlines() if json.loads(s)['split']==a.split]
    if a.limit: rows=rows[:a.limit]
    if a.base:
        from transformers import AutoModelForCausalLM
        model=AutoModelForCausalLM.from_pretrained(CACHE/'base',torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
    else: model=load_model(a.model,training=False,liger=False)
    start=time.monotonic(); outs=generate_rows(model,tok,rows,base=a.base)
    del model; gc.collect(); torch.cuda.empty_cache()
    def grade_code(pair):
        row,text=pair
        ok=code_correct(extract_code(text,row['prompt'],a.base,row['entry_point']),row['test'],row['entry_point'])
        return {**row,'completion':text,'correct':ok}
    pairs=list(zip(rows,outs)); scored=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        scored.extend(ex.map(grade_code,[p for p in pairs if p[0]['task']=='humaneval']))
    # math-verify uses signal alarms, which require each worker's main thread.
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as ex:
        scored.extend(ex.map(grade_math_pair,[p for p in pairs if p[0]['task']!='humaneval']))
    metrics={}
    for task in sorted({r['task'] for r in scored}):
        items=[r for r in scored if r['task']==task]; metrics[task]={'correct':sum(r['correct'] for r in items),'n':len(items),'accuracy':sum(r['correct'] for r in items)/len(items)}
    # Equal task weights prevent the large GSM8K split dominating selection.
    macro=sum(m['accuracy'] for m in metrics.values())/len(metrics)
    result={'model':a.model or 'base','base_prompting':a.base,'split':a.split,'metrics':metrics,'macro_accuracy':macro,'seconds':time.monotonic()-start,'seed':SEED,'max_new_tokens':1800,'effective_max_new_tokens':'min(1800, 2048 - padded_prompt_length)','context':2048,'prompt_token_cap':1100,'prompt_truncation':'left','generation':'greedy','eos_token_ids':[32001,0],'code_execution':'empty chroot, unprivileged UID, seccomp syscall allowlist, 3s CPU / 6s wall / 768MiB address space','math_grader':'math-verify'}
    profile=ROOT/'inference-profile.json'
    result['cache_strategy']='dynamic' if a.base or not profile.exists() else json.loads(profile.read_text()).get('selected',{}).get('mode','dynamic')
    result['evaluation_protocol']='v2-hashlib-entrypoint'
    result['code_extraction']='first complete code fence defining the requested entrypoint; supplied prompt context retained; no test-guided selection'
    result['sandbox_revision']='v2: hashlib and md5 preloaded before chroot; original syscall and resource limits retained'
    if result['cache_strategy']=='static': result['cache_detail']='compiled static KV cache, max length 2048; partial final batches use dynamic cache'
    write_json(a.output,result); Path(a.output).with_suffix('.samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    log('evaluation_complete',**result)

if __name__=='__main__': main()
