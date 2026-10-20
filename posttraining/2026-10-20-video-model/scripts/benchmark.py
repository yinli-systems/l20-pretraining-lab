import argparse, gc, json, statistics, time
import torch
from common import ROOT, setup_torch, load_model, log, write_json

def main():
    setup_torch(); model=load_model(training=True,liger=False); model.train()
    x=torch.randint(10,31000,(1,128),device='cuda'); y=x.clone(); y[:,:32]=-100
    with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(input_ids=x,labels=y).loss
    loss.backward(); expected=float(loss.detach()); checks={n:p.grad.detach().clone() for n,p in model.named_parameters() if n in ['model.layers.0.self_attn.q_proj.weight','model.layers.21.mlp.down_proj.weight']}
    model.zero_grad(set_to_none=True)
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    apply_liger_kernel_to_llama(model=model,rope=True,rms_norm=True,swiglu=True,fused_linear_cross_entropy=True,cross_entropy=False)
    with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(input_ids=x,labels=y).loss
    loss.backward(); actual=float(loss.detach()); equivalence={'reference_loss':expected,'optimized_loss':actual,'loss_abs_error':abs(expected-actual),'gradients':{}}
    for n,p in model.named_parameters():
        if n in checks:
            a=checks[n].float().flatten(); b=p.grad.float().flatten()
            cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0))
            rel=float((a-b).norm()/a.norm().clamp_min(1e-12))
            equivalence['gradients'][n]={'cosine':cosine,'relative_l2':rel}
            if cosine<0.98 or rel>0.15: raise RuntimeError(f'gradient equivalence failed: {n}')
    if abs(expected-actual)>0.03: raise RuntimeError('loss equivalence failed')
    write_json(ROOT/'kernel-equivalence.json',equivalence)
    del checks; model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-8,betas=(0.9,0.95),weight_decay=0.1,fused=True)
    results=[]
    def bench(fn,b,compiled):
        x=torch.randint(10,31000,(b,2048),device='cuda'); y=x.clone(); y[:,:512]=-100
        times=[]; torch.cuda.reset_peak_memory_stats()
        for step in range(10):
            torch.cuda.synchronize(); start=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16): loss=fn(input_ids=x,labels=y,use_cache=False).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,foreach=True); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); elapsed=time.monotonic()-start
            if step>=4: times.append(elapsed)
        r={'micro_batch':b,'sequence_length':2048,'compiled':compiled,'median_step_seconds':statistics.median(times),'tokens_per_second':b*2048/statistics.median(times),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'warmup_steps':4,'measured_steps':len(times)}
        results.append(r); log('benchmark_result',**r); write_json(ROOT/'throughput-benchmarks.json',results)
    for b in [4,8,12,16,24,32,40]:
        try: bench(model,b,False)
        except torch.OutOfMemoryError:
            log('benchmark_oom',batch=b); opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache(); break
    feasible=[r for r in results if r['peak_reserved_gib']<88]
    if not feasible: feasible=results
    best=max(feasible,key=lambda r:r['tokens_per_second'])
    try:
        compiled=torch.compile(model,dynamic=False,mode='default')
        bench(compiled,best['micro_batch'],True)
        del compiled
    except Exception as e:
        log('compiled_benchmark_failure',error=repr(e)); opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    chosen=max([r for r in results if r['peak_reserved_gib']<90] or results,key=lambda r:r['tokens_per_second'])
    # Reserve several GiB for the loader, validation and allocator fragmentation.
    write_json(ROOT/'selected-throughput.json',chosen)
    log('benchmark_complete',selected=chosen)

if __name__=='__main__': main()
