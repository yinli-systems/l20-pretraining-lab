"""One bounded controller. Wait for audited inputs; never infer readiness.

launch.json is materialized only after corpus/suite review. The controller may
wait on CPU, but its 48-hour clock starts before its first GPU preflight or
baseline, and every baseline, candidate, replication and confirmation counts.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

OLD = Path('/workspace/video-model')
JOB = OLD / 'validation-48h'
PREP_END = dt.datetime.fromisoformat('2026-10-20T19:48:39+00:00').timestamp()
child = None
RESUMING = False
LOADED_SOURCE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def status(phase, **extra):
    value = dict(status='running', phase=phase, utc=dt.datetime.now(dt.timezone.utc).isoformat(), **extra)
    write(JOB/'pipeline-status.json', value)
    print(json.dumps(value), flush=True)


def terminate(signum, frame):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=480)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    raise SystemExit(143)


def run(command, key, end):
    global child
    if time.time() > end-120:
        raise TimeoutError('Insufficient time remaining for '+key)
    status(key, command=command, child_deadline_unix=end)
    with (JOB/'logs'/f'{key}.log').open('a') as stream:
        child = subprocess.Popen(command, cwd=JOB, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            result = child.wait(timeout=max(1, end-time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=480)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise TimeoutError('Bounded stage expired: '+key)
        finally:
            child = None
    if result:
        raise RuntimeError(f'{key} exited {result}; inspect saved attempt, do not blindly resubmit')


def metric_valid(path, suite):
    value = json.loads(path.read_text())
    samples = [json.loads(x) for x in path.with_suffix('.samples.jsonl').read_text().splitlines()]
    ref = [json.loads(x) for x in Path(suite).read_text().splitlines()]
    if value['suite_sha256'] != sha(suite) or len(samples) != len(ref):
        raise RuntimeError('Evaluation snapshot mismatch')
    index = {r['id']: r for r in ref}
    if len(index) != len(ref) or len({r['id'] for r in samples}) != len(samples):
        raise RuntimeError('Duplicate evaluation records')
    for row in samples:
        reference = index[row['id']]
        if {k:v for k,v in row.items() if k not in ['completion','correct']} != reference or type(row['correct']) is not bool:
            raise RuntimeError('Evaluation row content mismatch')
    for domain, metric in value['metrics'].items():
        rows = [r for r in samples if r['domain'] == domain]
        if (sum(r['correct'] for r in rows), len(rows)) != (metric['correct'], metric['n']):
            raise RuntimeError('Evaluation count mismatch')
    return value


def evaluate(model, key, suite, frozen, end, retention=None, qwen=False):
    output = JOB/'evaluations'/f'{key}.json'
    command = [sys.executable, str(JOB/'evaluate_suite.py'), '--model', model,
               '--suite', suite, '--output', str(output), '--frozen-sources', frozen]
    if retention:
        command += ['--retention', retention]
    if qwen:
        command += ['--qwen']
    if not output.exists():
        run(command, key, end)
    return metric_valid(output, suite)


def guards(candidate, baseline):
    metrics = candidate['metrics']
    return (all(metrics[k]['accuracy'] >= baseline['metrics'][k]['accuracy']-.02 for k in metrics)
            and candidate['retention']['category_macro_accuracy'] >= baseline['retention']['category_macro_accuracy']-.02)


def train(config, end):
    stage = config['stage']
    config_path = JOB/'configs'/f'{stage}.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Cannot change a started stage configuration')
    write(config_path, config)
    completion = JOB/'stages'/stage/'complete.json'
    if not completion.exists():
        command=[sys.executable, str(JOB/'train_stage.py'), '--config', str(config_path)]
        recovery=Path('/dev/shm/video-model-validation48/recovery')/(stage+'.pt')
        if recovery.exists():
            if not RESUMING:
                raise RuntimeError('Existing stage recovery requires explicit bounded resume')
            command+=['--resume']
        run(command, 'train-'+stage, end)
    record = json.loads(completion.read_text())
    if record['config_identity'] != sha(config_path):
        raise RuntimeError('Completed training identity mismatch')
    # Endpoints and latest recovery must be verified locally before retiring
    # the previous optimizer snapshot or proceeding to another stage.
    status('awaiting_stage_backup', stage=stage)
    ack = JOB/'stage-backups'/f'{stage}.json'
    while time.time() < end:
        if ack.exists():
            value = json.loads(ack.read_text())
            if value.get('all_files_verified') and value['complete_sha256'] == sha(completion):
                # Storage policy approved for the existing small disks: keep
                # each pilot endpoint, but retire older completed optimizers
                # only after the latest full recovery is verified locally.
                active=Path(value['recovery']['checkpoint'])
                for old in active.parent.glob('*.pt'):
                    if old!=active and '.next.' not in old.name:
                        old.unlink()
                        old.with_suffix('.json').unlink(missing_ok=True)
                if stage in ['A-bridge','repeat-A-bridge']:
                    intermediate=stage.replace('bridge','cpt')
                    cpt_model=Path('/dev/shm/video-model-validation48/models')/intermediate
                    if cpt_model.exists():
                        shutil.rmtree(cpt_model)
                        write(JOB/'stages'/intermediate/'retired-intermediate.json',dict(
                            retained=False,reason='Bridge completed and latest full recovery verified locally; rolling storage policy',
                            bridge_stage=stage,utc=dt.datetime.now(dt.timezone.utc).isoformat()))
                return record
        time.sleep(10)
    raise TimeoutError('Local endpoint backup acknowledgement not received')


def main():
    global RESUMING
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    RESUMING=args.resume
    JOB.mkdir(parents=True, exist_ok=True)
    (JOB/'logs').mkdir(exist_ok=True)
    lock = (JOB/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    ready_path = JOB/'readiness.json'
    launch_path = JOB/'launch.json'
    budget_path = JOB/'budget.json'
    while not budget_path.exists() and time.time() < PREP_END and not (ready_path.exists() and launch_path.exists()
              and json.loads(ready_path.read_text()).get('ready')):
        status('waiting_for_audited_data_and_evaluation', gpu_budget_started=False)
        time.sleep(30)
    if not (launch_path.exists() and ready_path.exists() and json.loads(ready_path.read_text()).get('ready')):
        write(JOB/'pipeline-status.json', dict(status='stopped', reason='Preparation bound reached before readiness; no pilot GPU work launched'))
        return
    cfg = json.loads(launch_path.read_text())
    ready = json.loads(ready_path.read_text())
    if sha(__file__)!=LOADED_SOURCE_SHA:
        raise RuntimeError('Controller source changed while waiting; reload only before GPU budget starts')
    required = ['new_sft_corpus', 'new_cpt_corpus', 'suite_review', 'old_training_overlap_screen',
                'source_reviews_200_each', 'cpu_objective_parity', 'storage_and_local_monitor']
    if any(not ready['gates'].get(k) for k in required):
        raise RuntimeError('Audited readiness gates incomplete')
    for path, expected in ready['immutable_inputs'].items():
        if sha(path) != expected:
            raise RuntimeError('Readiness input changed: '+path)
    for key, n in [('dev',1200), ('confirmation',2400), ('retention',1000)]:
        if len(Path(cfg[key]).read_text().splitlines()) != n:
            raise RuntimeError('Unapproved custom suite size: '+key)
    comparison = json.loads((OLD/'comparisons/qwen-final-20261020/local-backup-verified.json').read_text())
    if not comparison.get('all_files_verified'):
        raise RuntimeError('Qwen comparison not yet backed up')
    now=time.time()
    if budget_path.exists():
        if not args.resume:
            raise RuntimeError('Existing budget: inspect attempt and explicit recovery; never reset clock')
        budget=json.loads(budget_path.read_text())
        if (budget['readiness_sha256']!=sha(ready_path) or budget['launch_sha256']!=sha(launch_path)
            or budget['deadline_unix']-budget['started_unix']!=48*3600 or now>=budget['deadline_unix']-7200):
            raise RuntimeError('Recovery cannot change frozen inputs or extend expired budget')
    else:
        if args.resume:
            raise RuntimeError('No existing budget to resume')
        budget = dict(started_unix=now, deadline_unix=now+48*3600, max_gpu_seconds=48*3600,
                  started_utc=dt.datetime.fromtimestamp(now,dt.timezone.utc).isoformat(),
                  deadline_utc=dt.datetime.fromtimestamp(now+48*3600,dt.timezone.utc).isoformat(),
                  readiness_sha256=sha(ready_path), launch_sha256=sha(launch_path),
                  includes='GPU preflight, all baselines, pilots, replication, confirmation and backup/shutdown reserve')
        write(budget_path, budget)
    end = budget['deadline_unix']
    # Preserve six hours for the once-only historical diagnostic of all arms,
    # independent confirmation, endpoint backup and instance stop.
    train_end = end-6*3600
    if not (JOB/'gpu-preflight.json').exists():
        run([sys.executable,str(JOB/'gpu_preflight.py'),'--launch',str(launch_path)], 'gpu-preflight', min(end,time.time()+1800))
    if not json.loads((JOB/'gpu-preflight.json').read_text())['passed']:
        raise RuntimeError('Native/fused parity or memory preflight failed')
    references = {}
    for name, model, qwen in [('start',cfg['sft_init'],False), ('qwen-0.5b',cfg['qwen_0_5'],True), ('qwen-1.5b',cfg['qwen_1_5'],True)]:
        references[name] = evaluate(model, name+'-dev', cfg['dev'], cfg['frozen_sources'],
                                    train_end, retention=cfg['retention'], qwen=qwen)
    baseline = references['start']
    candidates = {}
    for key in ['B1', 'B2']:
        record = train(cfg['stages'][key], train_end)
        candidates[key] = dict(model=record['model'], evaluation=evaluate(record['model'],key+'-dev',
                    cfg['dev'],cfg['frozen_sources'],train_end,retention=cfg['retention']))
    bridge_choice = max(['B1','B2'], key=lambda k:(guards(candidates[k]['evaluation'],baseline), candidates[k]['evaluation']['macro_accuracy']))
    cpt = train(cfg['stages']['A-cpt'], train_end)
    bridge = dict(cfg['stages']['A-bridge'], init=cpt['model'],
                  init_sha256={k:v['sha256'] for k,v in cpt['model_files'].items()},
                  lr=cfg['stages'][bridge_choice]['lr'])
    record = train(bridge, train_end)
    candidates['A'] = dict(model=record['model'], evaluation=evaluate(record['model'],'A-dev',
                cfg['dev'],cfg['frozen_sources'],train_end,retention=cfg['retention']))
    eligible = [k for k in candidates if guards(candidates[k]['evaluation'],baseline)]
    winner = max(eligible,key=lambda k:candidates[k]['evaluation']['macro_accuracy']) if eligible else None
    if winner and candidates[winner]['evaluation']['macro_accuracy'] <= baseline['macro_accuracy']:
        winner = None
    selection = dict(winner=winner or 'start', candidates=candidates, references=references,
                     selected_before_confirmation=True, second_seed='pending', bridge_lr_source=bridge_choice)
    write(JOB/'pilot-selection.json',selection)
    # Repeat only a winning recipe, at identical token/data/hyperparameter
    # budgets. If it cannot fit, report unreplicated, never claim promotion.
    if winner:
        required_seconds = cfg['replication_max_seconds'][winner]
        if time.time()+required_seconds < train_end:
            if winner == 'A':
                repeat_cpt = dict(cfg['stages']['A-cpt'],stage='repeat-A-cpt',seed=20261022)
                r = train(repeat_cpt,train_end)
                repeat = dict(bridge,stage='repeat-A-bridge',seed=20261022,init=r['model'],
                              init_sha256={k:v['sha256'] for k,v in r['model_files'].items()})
            else:
                repeat = dict(cfg['stages'][winner],stage='repeat-'+winner,seed=20261022)
            r = train(repeat,train_end)
            second = evaluate(r['model'],'second-seed-dev',cfg['dev'],cfg['frozen_sources'],train_end,retention=cfg['retention'])
            selection.update(second_seed='complete', second_seed_evaluation=second,
                             second_seed_same_direction=second['macro_accuracy']>baseline['macro_accuracy'])
        else:
            selection['second_seed']='not_run_insufficient_time_within_48h'
    else:
        selection['second_seed']='not_run_no_recipe_beat_guarded_start'
    selected_model = candidates[winner]['model'] if winner else cfg['sft_init']
    selection['selected_model']=selected_model
    write(JOB/'pilot-selection.json',selection)
    # The original exposed final split is now historical diagnosis only. Freeze
    # selection before these scores; never retune checkpoints from this output.
    historical=JOB/'historical-1791-diagnostic.jsonl'
    original=[json.loads(x) for x in (OLD/'benchmarks.jsonl').read_text().splitlines()]
    original=[dict(r,domain=r['task']) for r in original if r['split']=='final']
    if len(original)!=1791:
        raise RuntimeError('Historical comparison suite is not the original 1791 items')
    historical.write_text(''.join(json.dumps(r)+'\n' for r in original))
    historical_results={}
    for name,candidate in candidates.items():
        historical_results[name]=evaluate(candidate['model'],name+'-historical-1791',str(historical),
                                         cfg['frozen_sources'],end-4*3600)
    # First access to the held-out confirmation split occurs after selection.
    results = {}
    for name,model,qwen in [('start',cfg['sft_init'],False),('selected',selected_model,False),
                            ('qwen-0.5b',cfg['qwen_0_5'],True),('qwen-1.5b',cfg['qwen_1_5'],True)]:
        if name=='selected' and not winner:
            results[name]=results['start']
            continue
        results[name]=evaluate(model,name+'-confirmation',cfg['confirmation'],cfg['frozen_sources'],
                               end-3600,qwen=qwen)
    write(JOB/'final-report.json',dict(status='complete',selection=selection,confirmation=results,
                                      historical_diagnostics=historical_results,budget=budget))
    write(JOB/'pipeline-status.json',dict(status='complete',report=str(JOB/'final-report.json'),
               utc=dt.datetime.now(dt.timezone.utc).isoformat()))


if __name__=='__main__':
    try:
        main()
    except BlockingIOError:
        print('Existing controller holds the lock; no duplicate work started.',flush=True)
    except Exception as exc:
        write(JOB/'pipeline-status.json',dict(status='failed',error=repr(exc),
                   utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        raise
