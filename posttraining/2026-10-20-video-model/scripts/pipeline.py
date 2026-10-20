"""Bounded 24-hour experiment driver. All children are supervised and logged."""
import datetime, hashlib, json, os, shutil, signal, subprocess, sys, threading, time
from pathlib import Path
from common import ROOT,CACHE,write_json,log

START=datetime.datetime.fromisoformat('2026-10-19T19:48:39+00:00').timestamp()
END=START+24*3600
active=None; stop_requested=False

def terminate(signum=None,frame=None):
    global stop_requested
    stop_requested=True
    if active is not None and active.poll() is None:
        try: active.send_signal(signal.SIGTERM)
        except ProcessLookupError: pass

def run(name,args,limit=None,required=True):
    global active
    if stop_requested or time.time()>END-180: raise RuntimeError('24-hour budget exhausted')
    maximum=min(limit or (END-time.time()-180),END-time.time()-180)
    if maximum<=0: raise RuntimeError('No remaining budget')
    status={'phase':name,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_utc':datetime.datetime.fromtimestamp(END,datetime.timezone.utc).isoformat(),'command':args,'status':'running'}
    write_json(ROOT/'pipeline-status.json',status); log('phase_start',phase=name)
    with open(ROOT/'logs'/f'pipeline-{name}.log','a') as out:
        active=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        try: code=active.wait(timeout=maximum)
        except subprocess.TimeoutExpired:
            try: active.send_signal(signal.SIGTERM)
            except ProcessLookupError: pass
            try: code=active.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid,signal.SIGKILL); code=active.wait()
        finally:
            if stop_requested:
                try: os.killpg(active.pid,signal.SIGTERM)
                except ProcessLookupError: pass
            active=None
    status.update(status='complete' if code==0 else 'failed',exit_code=code,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(ROOT/'pipeline-status.json',status)
    with open(ROOT/'phase-history.jsonl','a') as f: f.write(json.dumps(status)+'\n')
    if required and code!=0: raise RuntimeError(f'phase {name} failed, exit {code}')
    return code==0

def main():
    signal.signal(signal.SIGTERM,terminate); signal.signal(signal.SIGINT,terminate)
    plan={'target':'English math and code reasoning','base':'AliceYin/L20-1B-20B-Base','budget_hours':24,'start_utc':datetime.datetime.fromtimestamp(START,datetime.timezone.utc).isoformat(),'hard_end_utc':datetime.datetime.fromtimestamp(END,datetime.timezone.utc).isoformat(),'sft':'full-parameter; FP32 master weights; BF16 forward/backward; compiled backbone; fused AdamW; assistant-only loss; 2048 complete-conversation packing','selection':'pilot learning-rate comparison on held-out loss, then independent development math/code accuracy; locked final set used once','rft':'four sampled answers per training prompt; mathematical answer checking or execution tests; train only complete successful responses; replay original data; retain only dev improvements','limits':'No global optimality or capability guarantee. Context remains native 2048. No publishing or paid teacher API calls.'}
    write_json(ROOT/'run-plan.json',plan)
    while not (CACHE/'packed/complete').exists():
        if time.time()>END-4*3600: raise RuntimeError('data preparation exceeded budget')
        time.sleep(10)
    run('preflight',[str(ROOT/'preflight.py')],limit=300)
    while not (ROOT/'refined-throughput-benchmarks.json').exists(): time.sleep(5)
    # The parent starts this only after the benchmark process has finished.
    if not (ROOT/'evaluations/base-dev.json').exists():
        run('baseline-dev',[str(ROOT/'evaluate.py'),'--base','--split','dev','--output',str(ROOT/'evaluations/base-dev.json')],limit=1800)
    pilots=[]
    for lr in [5e-5,1e-4,2e-4]:
        name='pilot-'+str(lr)
        completed=(ROOT/f'{name}-complete.json').exists() and (ROOT/f'{name}-result.json').exists() and json.loads((ROOT/f'{name}-result.json').read_text()).get('steps')==128
        if not completed:
            run(name,[str(ROOT/'train.py'),'--name',name,'--lr',str(lr),'--steps','128','--pilot','--evaluate-every','128','--deadline',str(min(END-4*3600,time.time()+1800))],limit=2100)
        result=json.loads((ROOT/f'{name}-result.json').read_text()); pilots.append({'name':name,**result})
    chosen=min(pilots,key=lambda x:x['validation_loss']); write_json(ROOT/'learning-rate-pilots.json',{'results':pilots,'selected':chosen})
    for pilot in pilots:
        if pilot['name']!=chosen['name']:
            shutil.rmtree(ROOT/'models'/pilot['name'],ignore_errors=True); (CACHE/f"{pilot['name']}-fp32.pt").unlink(missing_ok=True)
    if not (ROOT/'sft-complete.json').exists():
        # Preserve the original stage deadline across a checkpoint recovery.
        deadline_record=ROOT/'sft-deadline.json'
        if deadline_record.exists(): sft_end=min(END-8*3600,json.loads(deadline_record.read_text())['deadline'])
        else:
            sft_end=min(END-8*3600,time.time()+12*3600)
            write_json(deadline_record,{'deadline':sft_end})
        args=[str(ROOT/'train.py'),'--name','sft','--lr',str(chosen['lr']),'--epochs','3','--evaluate-every','2000','--deadline',str(sft_end)]
        if (ROOT/'resume-ready.json').exists() and json.loads((ROOT/'resume-ready.json').read_text()).get('stage')=='sft': args+=['--resume']
        else: args+=['--model',str(ROOT/'models'/chosen['name']),'--init-state',str(CACHE/f"{chosen['name']}-fp32.pt")]
        run('sft',args,limit=sft_end-time.time()+60)
    shutil.rmtree(ROOT/'models'/chosen['name'],ignore_errors=True)
    (CACHE/f"{chosen['name']}-fp32.pt").unlink(missing_ok=True)
    if not (ROOT/'models/sft-selected').exists(): shutil.copytree(ROOT/'models/best',ROOT/'models/sft-selected')
    stagnant_rounds=0
    for roundid in range(1,8):
        if time.time()>END-2.5*3600 or stop_requested: break
        if (ROOT/f'rft-{roundid}-complete.json').exists(): continue
        previous_selection=json.loads((ROOT/'best-model.json').read_text())
        gen_end=min(END-2*3600,time.time()+90*60)
        ready=ROOT/f'rft-{roundid}-ready.json'
        if not (ROOT/f'rft-{roundid}-complete.json').exists():
            run(f'rft-sampling-{roundid}',[str(ROOT/'rft.py'),'--round',str(roundid),'--deadline',str(gen_end)],limit=gen_end-time.time()+30,required=False)
            if ready.exists():
                r=json.loads(ready.read_text())
                run(f'rft-training-{roundid}',[str(ROOT/'train.py'),'--name',f'rft-{roundid}','--model',str(ROOT/'models/best'),'--data',r['data'],'--lr','1e-5','--epochs','2','--evaluate-every','100','--deadline',str(min(END-90*60,time.time()+45*60))],limit=2800,required=False)
            else: log('rft_round_unproductive',round=roundid)
        current_selection=json.loads((ROOT/'best-model.json').read_text())
        stagnant_rounds=0 if current_selection['dev_macro_accuracy']>previous_selection['dev_macro_accuracy'] else stagnant_rounds+1
        if stagnant_rounds>=2:
            log('rft_early_stop',reason='two verified RFT rounds without development-score improvement'); break
    # The final partition is not used to select or revise checkpoints.
    run('base-final',[str(ROOT/'evaluate.py'),'--base','--split','final','--output',str(ROOT/'evaluations/base-final.json')],limit=2400)
    run('best-final',[str(ROOT/'evaluate.py'),'--model',str(ROOT/'models/best'),'--split','final','--output',str(ROOT/'evaluations/best-final.json')],limit=2400)
    if ROOT.joinpath('models/candidate').exists(): shutil.rmtree(ROOT/'models/candidate')
    hashes={}
    for p in (ROOT/'models/best').iterdir():
        if p.is_file():
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
            hashes[p.name]=h.hexdigest()
    write_json(ROOT/'final-model-sha256.json',hashes)
    base=json.loads((ROOT/'evaluations/base-final.json').read_text()); best=json.loads((ROOT/'evaluations/best-final.json').read_text())
    write_json(ROOT/'final-report.json',{'status':'complete','plan':plan,'selected_checkpoint':str(ROOT/'models/best'),'base':base,'posttrained':best,'learning_rate':chosen,'selection':json.loads((ROOT/'best-model.json').read_text()),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    write_json(ROOT/'pipeline-status.json',{'status':'complete','model':str(ROOT/'models/best'),'report':str(ROOT/'final-report.json')}); log('pipeline_complete')

if __name__=='__main__':
    try: main()
    except BaseException as e:
        was_stopping=stop_requested
        terminate(); write_json(ROOT/'pipeline-status.json',{'status':'stopped' if was_stopping else 'failed','error':repr(e),'time':time.time(),'best_checkpoint':str(ROOT/'models/best')}); raise
