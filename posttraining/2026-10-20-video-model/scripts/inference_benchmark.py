"""Compare dynamic and compiled static KV caches before verified RFT rollouts."""
import argparse,gc,json,statistics,time
import torch
from transformers import StaticCache,CompileConfig
from common import ROOT,setup_torch,load_model,tokenizer,write_json,log
from evaluate import prompt_text

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); a=p.parse_args()
    setup_torch(); model=load_model(a.model,training=False,liger=False); model.eval(); tok=tokenizer(); tok.padding_side='left'; tok.truncation_side='left'
    rows=[json.loads(s) for s in (ROOT/'benchmarks.jsonl').read_text().splitlines() if json.loads(s)['split']=='dev'][:32]
    inputs=tok([prompt_text(tok,r) for r in rows],padding=True,truncation=True,max_length=1100,add_special_tokens=False,return_tensors='pt').to('cuda'); inputs.pop('token_type_ids',None)
    settings=dict(max_new_tokens=128,min_new_tokens=128,do_sample=False,use_cache=True,eos_token_id=[32001,0],pad_token_id=tok.pad_token_id)
    results=[]; reference=None
    for mode in ['dynamic','static']:
        cache=StaticCache(config=model.config,max_cache_len=2048) if mode=='static' else None
        times=[]; agreement=[]
        try:
            for i in range(4):
                if cache is not None: cache.reset()
                torch.cuda.synchronize(); start=time.monotonic()
                extra={'past_key_values':cache,'compile_config':CompileConfig(dynamic=False,mode='reduce-overhead')} if cache is not None else {}
                with torch.inference_mode(): output=model.generate(**inputs,**settings,**extra)
                torch.cuda.synchronize(); elapsed=time.monotonic()-start
                if i>=1: times.append(elapsed)
                continuation=output[:,inputs.input_ids.shape[1]:]
                if reference is None: reference=continuation.detach().clone()
                agreement.append(float((continuation==reference).float().mean()))
            r={'mode':mode,'tokens_per_second':32*128/statistics.median(times),'median_batch_seconds':statistics.median(times),'greedy_token_agreement':min(agreement),'measured_batches':3,'fixed_generated_tokens':128,'cache_max_length':2048}
            results.append(r); log('inference_benchmark',**r)
        except Exception as e: log('inference_benchmark_failure',mode=mode,error=repr(e))
        del cache; gc.collect(); torch.cuda.empty_cache()
    eligible=[r for r in results if r['greedy_token_agreement']>=0.9999]
    selected=max(eligible,key=lambda r:r['tokens_per_second']) if eligible else {'mode':'dynamic'}
    write_json(ROOT/'inference-profile.json',{'selected':selected,'benchmarks':results,'scope':'32 development prompts, forced 128-token greedy continuations; selected only with >=99.99% token agreement; first warmup batch excluded'})

if __name__=='__main__': main()
