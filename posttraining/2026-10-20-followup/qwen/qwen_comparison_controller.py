"""One bounded, supervised comparator job; original training/backup stay intact."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/workspace/video-model')
JOB = ROOT/'comparisons/qwen-final-20261020'
END = dt.datetime.fromisoformat('2026-10-20T19:48:39+00:00').timestamp()
MODELS = [('qwen-0.5b','Qwen/Qwen2.5-0.5B-Instruct'),
          ('qwen-1.5b','Qwen/Qwen2.5-1.5B-Instruct')]
child = None


def write(name, data):
    p = JOB/name
    t = p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(data,indent=2)+'\n'); t.replace(p)


def status(phase, **kw):
    d = {'phase':phase,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),**kw}
    write('status.json',d); print(json.dumps(d),flush=True)


def stop(signum, frame):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
    raise SystemExit('comparison terminated')


def main():
    global child
    JOB.mkdir(parents=True,exist_ok=True)
    if (JOB/'complete.json').exists():
        raise RuntimeError('Comparison already complete; refusing duplicate generation')
    if time.time() > END-90*60:
        raise RuntimeError('Insufficient time in original 24-hour budget')
    assert json.loads((ROOT/'final-report.json').read_text())['status']=='complete'
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    import torch
    # This is the active evaluation controller's CUDA context. The existing
    # stop_instance.py GPU-process check protects downloads, model handoffs and
    # local-result acknowledgement without stopping the existing backup worker.
    guard = torch.zeros(1,device='cuda')
    torch.cuda.synchronize()
    status('preparing',pid=os.getpid(),original_budget_end_utc='2026-10-20T19:48:39+00:00')
    from huggingface_hub import HfApi,snapshot_download
    models = []
    prior = JOB/'plan.json'
    locked = json.loads(prior.read_text()) if prior.exists() else None
    for key,repo in MODELS:
        revision = next(x['revision'] for x in locked['models'] if x['key']==key) if locked else HfApi().model_info(repo).sha
        directory = Path('/dev/shm/l20-posttrain/qwen-comparators')/key
        models.append({'key':key,'repo':repo,'revision':revision,'directory':str(directory)})
    protocol = json.loads((ROOT/'evaluations/best-final.json').read_text())
    plan = {'job_id':'qwen-final-20261020-v1','models':models,
            'requested_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'budget_end_utc':'2026-10-20T19:48:39+00:00',
            'benchmark_sha256':hashlib.sha256((ROOT/'benchmarks.jsonl').read_bytes()).hexdigest(),
            'frozen_sources':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['evaluate.py','common.py','sandbox.py']},
            'reference_protocol':protocol,
            'adaptation':'Same user instructions and frozen generator/grader; native tokenizer/chat template and semantic native EOS IDs. Qwen template may add its default system message. No sampling, quantization, training, or test-guided prompt changes.'}
    write('plan.json',plan)
    try:
        for model in models:
            name = model['key']
            if (JOB/(name+'.json')).exists():
                raise RuntimeError('Existing result; inspect before attempting any generation')
            status('downloading',model=name,revision=model['revision'])
            snapshot_download(model['repo'],revision=model['revision'],local_dir=model['directory'],
                              allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json','tokenizer.model'],max_workers=4)
            status('evaluating',model=name,revision=model['revision'])
            with (JOB/(name+'.log')).open('a') as log:
                child = subprocess.Popen([sys.executable,str(JOB/'qwen_comparison_worker.py'),
                    '--key',name,'--model',model['directory'],'--revision',model['revision']],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                timeout = min(5400,END-time.time()-600)
                if timeout<=0: raise RuntimeError('Original budget exhausted')
                try: code=child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
                    raise RuntimeError('Comparator timed out within original budget')
                child=None
                if code: raise RuntimeError(name+' exited '+str(code))
        files={}
        for p in sorted(JOB.iterdir()):
            if p.is_file() and p.name not in ['status.json','complete.json','local-backup-verified.json'] and not p.name.endswith('.tmp'):
                raw=p.read_bytes();files[p.name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        write('complete.json',{'status':'complete','finished_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'files':files})
        status('awaiting_local_backup',complete=True)
    except Exception as exc:
        status('failed',error=repr(exc),pid=os.getpid())
    # Keep the evaluation task present until its small result files have been
    # copied and hash verified locally, or the unchanged hard budget is reached.
    while time.time()<END-180:
        ack=JOB/'local-backup-verified.json'
        if ack.exists() and json.loads(ack.read_text()).get('all_files_verified'):
            status('released_for_autostop',local_result_backup_verified=True)
            del guard; return
        time.sleep(10)
    status('original_budget_reached',local_result_backup_verified=False)


if __name__=='__main__':
    main()
