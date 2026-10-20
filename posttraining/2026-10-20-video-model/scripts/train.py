"""Full-parameter BF16 mixed-precision SFT with FP32 master weights and AdamW."""
import argparse, contextlib, gc, json, math, os, shutil, signal, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from common import ROOT, CACHE, SEED, setup_torch, load_model, tokenizer, write_json, log, json_safe

class Packed(Dataset):
    def __init__(self,split='train',directory=None):
        p=Path(directory or CACHE/'packed'); self.ids=np.memmap(p/f'{split}.ids.u16',mode='r',dtype=np.uint16).reshape(-1,2048); self.mask=np.memmap(p/f'{split}.mask.u8',mode='r',dtype=np.uint8).reshape(-1,2048)
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        x=torch.from_numpy(np.array(self.ids[i],dtype=np.int64)); m=torch.from_numpy(np.array(self.mask[i],dtype=np.bool_))
        y=x.clone(); y[~m]=-100
        return x,y,int(m[1:].sum())

class EpochBatches:
    def __init__(self,n,b,consumed=0): self.n=n; self.b=b; self.per_epoch=n//b; self.consumed=consumed
    def __len__(self): return self.per_epoch
    def __iter__(self):
        epoch,offset=divmod(self.consumed,self.per_epoch)
        order=np.random.default_rng(SEED+epoch).permutation(self.n)
        for j in range(offset,self.per_epoch): yield order[j*self.b:(j+1)*self.b].tolist()

def loader(ds,batch,shuffle=True):
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=2,pin_memory=True,persistent_workers=True,drop_last=shuffle,generator=torch.Generator().manual_seed(SEED),prefetch_factor=3)

def training_action(step,total,evaluate_every,pilot=False,finishing=False):
    if step%evaluate_every==0 or step==total or finishing: return 'evaluate'
    if not pilot and (step==20 or step%500==0): return 'checkpoint'
    return None

def export(model,dest):
    dest=Path(dest); temp=dest.with_name(dest.name+'-next'); old=dest.with_name(dest.name+'-old')
    if temp.exists(): shutil.rmtree(temp)
    state={k:v.detach().to(device='cpu',dtype=torch.bfloat16).contiguous() for k,v in model.state_dict().items()}
    model.save_pretrained(temp,state_dict=state,safe_serialization=True,max_shard_size='2GB')
    tokenizer().save_pretrained(temp)
    del state
    if old.exists(): shutil.rmtree(old)
    if dest.exists(): dest.rename(old)
    temp.rename(dest)
    if old.exists(): shutil.rmtree(old)

def save_resume(model,opt,state):
    target=ROOT/'resume.pt'; ready=ROOT/'resume-ready.json'; staging=CACHE/'resume-next.pt'
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'state':state,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},staging)
    ready.unlink(missing_ok=True)
    shutil.copyfile(staging,target); staging.unlink()
    write_json(ready,{'step':state['step'],'bytes':target.stat().st_size,'time':time.time(),'stage':state['stage']})

@torch.no_grad()
def validation(model,dl,max_batches=24):
    model.eval(); total=0.; n=0
    for i,(x,y,count) in enumerate(dl):
        if i>=max_batches: break
        x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(input_ids=x,labels=y,use_cache=False,skip_logits=True).loss
        c=int(count.sum()); total+=float(loss)*c; n+=c
    model.train()
    return total/max(1,n)

def run_eval(model,path,tag,timeout=1200):
    gc.collect(); torch.cuda.empty_cache()
    out=ROOT/'evaluations'/f'{tag}.json'; out.parent.mkdir(exist_ok=True)
    command=[sys.executable,str(ROOT/'evaluate.py'),'--model',str(path),'--split','dev','--output',str(out)]
    with open(ROOT/'logs'/f'eval-{tag}.log','w') as f:
        result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode!=0: raise RuntimeError(f'evaluation failed: {tag}; inspect log')
    return json.loads(out.read_text())

def main():
    p=argparse.ArgumentParser(); p.add_argument('--name',required=True); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--steps',type=int,default=0); p.add_argument('--epochs',type=float,default=3); p.add_argument('--deadline',type=float,required=True); p.add_argument('--model'); p.add_argument('--init-state'); p.add_argument('--data'); p.add_argument('--pilot',action='store_true'); p.add_argument('--evaluate-every',type=int,default=2000); p.add_argument('--resume',action='store_true'); a=p.parse_args()
    if a.name.startswith('rft-') and a.data:
        if a.resume:
            quality_record=ROOT/f'{a.name}-quality-gate.json'
            if quality_record.exists():
                quality=json.loads(quality_record.read_text())
                if quality.get('status')!='ready': raise RuntimeError('RFT quality dataset is not ready for resume')
                a.data=quality['data']
        else:
            from rft_quality import prepare_quality_dataset
            cleaned=prepare_quality_dataset(a.name,a.data)
            if cleaned is None:
                write_json(ROOT/f'{a.name}-complete.json',{'stage':a.name,'step':0,'skipped':'fewer than 256 quality-checked prompts','evaluation_protocol':'v2-hashlib-entrypoint'})
                log('training_skipped',stage=a.name,reason='RFT quality gate')
                return
            a.data=str(cleaned)
    if a.name=='sft' and (ROOT/'advanced-enabled.json').exists():
        wait_until=min(a.deadline-3600,time.time()+1800)
        while not (CACHE/'packed/advanced-complete').exists() and time.time()<wait_until: time.sleep(5)
        if not (CACHE/'packed/advanced-complete').exists(): log('advanced_data_wait_expired',fallback='original audited data')
    if a.name=='sft':
        if not a.resume:
            with open(ROOT/'logs/main-sft-preflight.log','w') as f:
                subprocess.run([sys.executable,str(ROOT/'preflight.py')],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        baseline=ROOT/'evaluations/base-dev.json'
        if not baseline.exists() or json.loads(baseline.read_text()).get('max_new_tokens')!=1800:
            write_json(ROOT/'evaluation-protocol.json',{'context':2048,'max_new_tokens':1800,'effective_max_new_tokens':'min(1800, 2048 - padded_prompt_length)','prompt_token_cap':1100,'generation':'greedy','eos_token_ids':[32001,0],'frozen_before_main_sft':True,'reason':'Allow complete reasoning within the native context rather than a 768-token output ceiling.'})
            with open(ROOT/'logs/base-dev-native-context.log','w') as f:
                subprocess.run([sys.executable,str(ROOT/'evaluate.py'),'--base','--split','dev','--output',str(baseline)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1500)
    setup_torch(); cfg=json.loads((ROOT/'selected-throughput.json').read_text()); b=cfg['micro_batch']; accum=max(1,math.ceil(65536/(b*2048)))
    ds=Packed(directory=a.data); sampler=EpochBatches(len(ds),b)
    dl=DataLoader(ds,batch_sampler=sampler,num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=3)
    vd=loader(Packed('valid',directory=a.data),min(4,b),False)
    steps=a.steps or math.ceil(len(dl)*a.epochs/accum)
    model=load_model(a.model,training=True); model.train()
    if a.init_state: model.load_state_dict(torch.load(a.init_state,map_location='cpu',weights_only=True))
    if cfg.get('compiled'): model.model.forward=torch.compile(model.model.forward,dynamic=False,mode='default')
    params=[{'params':[v for n,v in model.named_parameters() if v.ndim>=2],'weight_decay':0.1},{'params':[v for n,v in model.named_parameters() if v.ndim<2],'weight_decay':0.0}]
    opt=torch.optim.AdamW(params,lr=a.lr,betas=(0.9,0.95),eps=1e-8,fused=True)
    step=0; tokens=0; best_loss=float('inf'); best_score=-1; stale=0; warmup=max(8,int(steps*0.03)); stop=False
    prior_stage=None
    if not a.pilot and (ROOT/'best-model.json').exists():
        prior=json.loads((ROOT/'best-model.json').read_text()); best_score=prior['dev_macro_accuracy']; prior_stage=prior['stage']
    state={'step':0,'stage':a.name,'tokens':0,'lr':a.lr,'micro_batch':b,'gradient_accumulation':accum,'planned_steps':steps}
    state['training_data_directory']=str(Path(a.data or CACHE/'packed'))
    if a.name.startswith('rft-') and (ROOT/f'{a.name}-quality-gate.json').exists():
        state['training_data_quality_revision']=json.loads((ROOT/f'{a.name}-quality-gate.json').read_text())['revision']
    if a.resume:
        ckpt=torch.load(ROOT/'resume.pt',map_location='cuda',weights_only=False)
        if ckpt['state']['stage']!=a.name: raise RuntimeError('resume stage mismatch')
        if 'training_data_directory' in ckpt['state'] and ckpt['state']['training_data_directory']!=state['training_data_directory']: raise RuntimeError('resume data directory mismatch')
        model.load_state_dict(ckpt['model']); opt.load_state_dict(ckpt['optimizer']); state=ckpt['state']; step=state['step']; tokens=state['tokens']; best_loss=state.get('best_loss',best_loss); best_score=state.get('best_score',best_score)
        sampler.consumed=state.get('micro_batches_consumed',step*accum)
        torch.set_rng_state(ckpt['rng'].cpu()); torch.cuda.set_rng_state(ckpt['cuda_rng'].cpu()); del ckpt
        # A recorded grader bug fix can rescore the existing selected checkpoint.
        # Use that verified score rather than the old scalar embedded in resume.pt.
        if not a.pilot and (ROOT/'best-model.json').exists():
            prior=json.loads((ROOT/'best-model.json').read_text())
            if prior.get('evaluation_protocol')=='v2-hashlib-entrypoint':
                best_score=prior['dev_macro_accuracy']; prior_stage=prior['stage']
                state['best_score']=best_score
                state['evaluation_protocol']=prior['evaluation_protocol']
    last_saved_step=step if a.resume else -1
    def stopped(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopped); signal.signal(signal.SIGINT,stopped)
    it=iter(dl); started=time.monotonic(); window=time.monotonic(); window_tokens=0; window_supervised=0; loss_sum=0.; interval=0
    metricsfile=open(ROOT/'logs'/f'{a.name}-metrics.jsonl','a',buffering=1)
    backend=cfg.get('attention_backend','auto')
    from torch.nn.attention import sdpa_kernel,SDPBackend
    ctx=sdpa_kernel(SDPBackend.CUDNN_ATTENTION) if backend=='cudnn' else sdpa_kernel(SDPBackend.FLASH_ATTENTION) if backend=='flash' else contextlib.nullcontext()
    with ctx:
        while step<steps and time.time()<a.deadline-120 and not stop:
            batches=[]
            for _ in range(accum):
                try: batch=next(it)
                except StopIteration: it=iter(dl); batch=next(it)
                batches.append(batch)
                sampler.consumed+=1
            total=sum(int(q[2].sum()) for q in batches)
            schedule=(step+1)/warmup if step<warmup else 0.1+0.9*0.5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,steps-warmup))))
            lr=a.lr*schedule
            for g in opt.param_groups: g['lr']=lr
            batchloss=0.; supervised=0
            for x,y,c in batches:
                weight=int(c.sum())/total; x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
                with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(input_ids=x,labels=y,use_cache=False).loss
                (loss*weight).backward(); batchloss+=float(loss.detach())*weight; supervised+=int(c.sum())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,foreach=True)
            if not torch.isfinite(norm) or not math.isfinite(batchloss): raise RuntimeError('nonfinite loss or gradient; preserving previous best model')
            opt.step(); opt.zero_grad(set_to_none=True); step+=1
            batchtokens=b*2048*accum; tokens+=batchtokens; window_tokens+=batchtokens; window_supervised+=supervised; loss_sum+=batchloss; interval+=1
            state.update(step=step,tokens=tokens,best_loss=best_loss,best_score=best_score,micro_batches_consumed=sampler.consumed)
            if step%20==0 or step==1:
                torch.cuda.synchronize(); elapsed=time.monotonic()-window
                stats={**state,'loss':loss_sum/interval,'lr_current':lr,'gradient_norm':float(norm),'packed_tokens_per_second':window_tokens/elapsed,'supervised_tokens_per_second':window_supervised/elapsed,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'elapsed_seconds':time.monotonic()-started}
                metricsfile.write(json.dumps(json_safe(stats),allow_nan=False)+'\n'); write_json(ROOT/'training-status.json',stats); log('training',**stats)
                window=time.monotonic(); window_tokens=0; window_supervised=0; loss_sum=0.; interval=0
            finishing=stop or time.time()>=a.deadline-240
            action=training_action(step,steps,a.evaluate_every,a.pilot,finishing)
            if action=='evaluate':
                val=validation(model,vd); log('validation',stage=a.name,step=step,loss=val)
                if a.pilot:
                    export(model,ROOT/'models'/a.name)
                    torch.save(model.state_dict(),CACHE/f'{a.name}-fp32.pt')
                    write_json(ROOT/f'{a.name}-result.json',{'lr':a.lr,'validation_loss':val,'steps':step,'tokens':tokens})
                else:
                    improved_loss=val<best_loss; best_loss=min(best_loss,val)
                    if not stop: export(model,ROOT/'models'/'candidate')
                    if not stop and time.time()<a.deadline-900:
                        ev=run_eval(model,ROOT/'models'/'candidate',f'{a.name}-{step}',timeout=min(1200,int(a.deadline-time.time()-120)))
                        score=ev['macro_accuracy']
                        if score>best_score or (score==best_score and improved_loss and prior_stage in [None,a.name]):
                            best_score=score; stale=0; export(model,ROOT/'models'/'best')
                            prior_stage=a.name
                            write_json(ROOT/'best-model.json',{'stage':a.name,'step':step,'validation_loss':val,'dev_macro_accuracy':score,'evaluation':str(ROOT/'evaluations'/f'{a.name}-{step}.json'),'evaluation_protocol':ev.get('evaluation_protocol')})
                        else: stale+=1
                    elif not (ROOT/'models/best').exists(): export(model,ROOT/'models'/'best')
                    state.update(best_loss=best_loss,best_score=best_score)
                    save_resume(model,opt,state)
                    last_saved_step=step
                    if stale>=3: log('early_stop',reason='three development evaluations without improvement'); break
                window=time.monotonic(); window_tokens=0; window_supervised=0; loss_sum=0.; interval=0
                if finishing: break
            elif action=='checkpoint':
                val=validation(model,vd); log('checkpoint_validation',stage=a.name,step=step,loss=val)
                save_resume(model,opt,state)
                last_saved_step=step
                window=time.monotonic(); window_tokens=0; window_supervised=0; loss_sum=0.; interval=0
    if not a.pilot:
        if not (ROOT/'models/best').exists(): export(model,ROOT/'models'/'best')
        if last_saved_step!=step: save_resume(model,opt,state)
    metricsfile.close()
    write_json(ROOT/f"{a.name}-{'interrupted' if stop else 'complete'}.json",state)
    log('training_interrupted' if stop else 'training_complete',**state)
    if stop: raise SystemExit(143)

if __name__=='__main__': main()
