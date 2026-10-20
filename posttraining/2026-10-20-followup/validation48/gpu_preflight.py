"""Actual-model BF16 fused/native gradient parity and training memory preflight.

Invoked only after the driver starts the inclusive 48h GPU budget. Disposable
benchmark weights are updated; all real pilot stages reload their fixed init.
"""
import argparse
import contextlib
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM
from objective import Packed

OLD=Path('/workspace/video-model')
JOB=OLD/'validation-48h'
sys.path.insert(0,str(OLD))
from common import load_model,write_json


def sample(directory,objective):
    data=Packed(directory,objective)
    examples=[]
    for i in range(2):
        ids,labels,count=data[i]
        target=torch.nonzero(labels!=-100).flatten()
        begin=max(0,min(2048-256,int(target[0])-64))
        examples.append((ids[begin:begin+256],labels[begin:begin+256]))
    return torch.stack([r[0] for r in examples]).cuda(),torch.stack([r[1] for r in examples]).cuda()


def gradients(model,x,y):
    model.train()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result=model(input_ids=x,labels=y,use_cache=False)
    loss=float(result.loss.detach())
    result.loss.backward()
    grad={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
    return loss,grad


def parity(cfg):
    inputs={objective:sample(cfg['stages'][key]['data'],objective)
            for objective,key in [('sft','B1'),('cpt','A-cpt')]}
    references={}
    native=AutoModelForCausalLM.from_pretrained(cfg['sft_init'],torch_dtype=torch.float32,
                                             attn_implementation='sdpa',trust_remote_code=False).cuda()
    for objective,(x,y) in inputs.items():
        native.zero_grad(set_to_none=True)
        references[objective]=gradients(native,x,y)
    del native
    gc.collect();torch.cuda.empty_cache()
    fused=load_model(cfg['sft_init'],training=True)
    results={}
    for objective,(x,y) in inputs.items():
        fused.zero_grad(set_to_none=True)
        loss,grad=gradients(fused,x,y)
        reference_loss,reference_grad=references[objective]
        square=norm=dot=other_norm=0.
        if grad.keys()!=reference_grad.keys():
            raise RuntimeError('Native/fused trainable parameter gradient sets differ')
        for n in grad:
            a,b=grad[n].float(),reference_grad[n].float()
            square+=float((a-b).square().sum())
            norm+=float(b.square().sum())
            other_norm+=float(a.square().sum())
            dot+=float((a*b).sum())
        rel=(square/max(norm,1e-30))**.5
        cosine=dot/max((norm*other_norm)**.5,1e-30)
        value=dict(native_loss=reference_loss,fused_loss=loss,
                   relative_gradient_l2=rel,gradient_cosine=cosine,
                   passed=abs(loss-reference_loss)<=.02 and rel<=.03 and cosine>=.999)
        results[objective]=value
        if not value['passed']:
            write_json(JOB/'gpu-preflight.json',dict(passed=False,parity=results,
                         action='Inspect implementation; do not relax tolerances after observing scores.'))
            raise RuntimeError('Native/fused actual-model parity failed for '+objective)
    del references,reference_grad,grad,inputs
    fused.zero_grad(set_to_none=True)
    gc.collect();torch.cuda.empty_cache()
    return fused,results


def benchmark(model,cfg):
    ds={objective:Packed(cfg['stages'][key]['data'],objective)
        for objective,key in [('sft','B1'),('cpt','A-cpt')]}
    # The previous same-GPU study selected microbatch22; validate on the actual
    # new dense CPT and sparse SFT objectives, with Adam states materialized.
    model.model.forward=torch.compile(model.model.forward,dynamic=False,mode='default')
    opt=torch.optim.AdamW(model.parameters(),lr=1e-7,betas=(.9,.95),fused=True)
    from torch.nn.attention import sdpa_kernel,SDPBackend
    profiles={}
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        for objective in ['cpt','sft']:
            b=22
            if len(ds[objective])<b*2:
                raise RuntimeError('Preflight dataset too small')
            batches=[]
            for j in range(2):
                rows=[ds[objective][j*b+i] for i in range(b)]
                batches.append((torch.stack([r[0] for r in rows]).cuda(),
                                torch.stack([r[1] for r in rows]).cuda(),sum(r[2] for r in rows)))
            elapsed=[]
            torch.cuda.reset_peak_memory_stats()
            for step in range(7):
                torch.cuda.synchronize();start=time.monotonic()
                opt.zero_grad(set_to_none=True)
                total=sum(batch[2] for batch in batches)
                for x,y,n in batches:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=model(input_ids=x,labels=y,use_cache=False).loss
                    (loss*n/total).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
                opt.step()
                torch.cuda.synchronize()
                if step>=2:
                    elapsed.append(time.monotonic()-start)
            free,total_memory=torch.cuda.mem_get_info()
            profiles[objective]=dict(microbatch=b,accumulation=2,
                median_packed_tokens_per_second=b*2*2048/statistics.median(elapsed),
                median_supervised_tokens_per_second=total/statistics.median(elapsed),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),measured_steps=5,
                optimizer_states_materialized=True,free_bytes_after_benchmark=free)
            del batches
            gc.collect();torch.cuda.empty_cache()
    return profiles


def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',required=True);args=p.parse_args()
    cfg=json.loads(Path(args.launch).read_text())
    budget=json.loads((JOB/'budget.json').read_text())
    if time.time()>=budget['deadline_unix']:
        raise RuntimeError('No unbudgeted GPU preflight allowed')
    torch.set_num_threads(8)
    torch.manual_seed(20261020)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32=True
    torch._inductor.config.compile_threads=8
    model,checks=parity(cfg)
    profile=benchmark(model,cfg)
    write_json(JOB/'gpu-preflight.json',dict(passed=True,parity=checks,throughput=profile,
        selected=dict(micro_batch=22,accumulation=2,compiled=True,attention_backend='cudnn'),
        interpretation='Measured actual new objectives at prior selected batch; not a claim of global throughput optimality.'))


if __name__=='__main__':
    main()
