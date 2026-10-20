import gc,json,statistics,time,contextlib
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from common import ROOT,setup_torch,load_model,write_json,log

def main():
    setup_torch(); model=load_model(training=True); model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-9,betas=(0.9,0.95),weight_decay=0.1,fused=True)
    original=model.model.forward
    results=[]
    configs=[(16,False,'auto'),(20,False,'auto'),(16,True,'auto'),(20,True,'auto'),(22,True,'auto'),(20,True,'cudnn'),(20,True,'flash')]
    for b,compiled,attn in configs:
        model.model.forward=torch.compile(original,dynamic=False,mode='default') if compiled else original
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            x=torch.randint(10,31000,(b,2048),device='cuda'); y=x.clone(); y[:,:512]=-100
            times=[]
            ctx=sdpa_kernel(SDPBackend.CUDNN_ATTENTION) if attn=='cudnn' else sdpa_kernel(SDPBackend.FLASH_ATTENTION) if attn=='flash' else contextlib.nullcontext()
            with ctx:
                for s in range(11):
                    # Vary the supervision count to expose unwanted loss recompiles.
                    y[:,512]=x[:,512] if s%2 else -100
                    torch.cuda.synchronize(); start=time.monotonic()
                    with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(input_ids=x,labels=y,use_cache=False).loss
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,foreach=True); opt.step(); opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    if s>=4: times.append(time.monotonic()-start)
            r={'micro_batch':b,'sequence_length':2048,'compiled':compiled,'compile_scope':'backbone','attention_backend':attn,'tokens_per_second':b*2048/statistics.median(times),'median_step_seconds':statistics.median(times),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'measured_steps':len(times)}
            results.append(r); write_json(ROOT/'refined-throughput-benchmarks.json',results); log('refined_benchmark',**r)
            del x,y,loss
        except Exception as e:
            log('refined_failure',batch=b,compiled=compiled,attention=attn,error=repr(e)); opt.zero_grad(set_to_none=True)
            try: del x,y,loss
            except UnboundLocalError: pass
            gc.collect(); torch.cuda.empty_cache()
    eligible=[r for r in results if r['peak_allocated_gib']<89]
    chosen=max(eligible,key=lambda x:x['tokens_per_second'])
    write_json(ROOT/'selected-throughput.json',chosen); log('refined_selection',**chosen)

if __name__=='__main__': main()
