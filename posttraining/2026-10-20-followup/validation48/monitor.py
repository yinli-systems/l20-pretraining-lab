"""Single local monitor for preparation -> authorized pilots -> verified stop.

The old monitor completed its historical backup and exited; it must not be
restarted against an already-complete old pipeline during new stage handoffs.
"""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

LOCAL = Path(__file__).resolve().parents[1]
ART = LOCAL / 'artifacts/validation48'
DEST = ART / 'remote-receipts'
REMOTE = '/workspace/video-model/validation-48h'
WORK = '/dev/shm/video-model-validation48'
SSH_TARGET = os.environ.get('VIDEO_MODEL_SSH_TARGET', '')
SSH_PORT = str(int(os.environ.get('VIDEO_MODEL_SSH_PORT', '22')))
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', SSH_PORT, SSH_TARGET]
PREP_END = dt.datetime.fromisoformat('2026-10-20T19:48:39+00:00').timestamp()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def event(name, **extra):
    row = dict(event=name, utc=dt.datetime.now(dt.timezone.utc).isoformat(), **extra)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    write(ART / 'monitor-status.json', row)


def ssh(command, timeout=60, input=None):
    result = subprocess.run(SSH + [command], input=input, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError('Remote operation failed: ' + result.stderr[-500:])
    return result.stdout


def remote_state():
    script = '''import pathlib,json
r=pathlib.Path('/workspace/video-model/validation-48h')
names=['preparation-status.json','readiness.json','budget.json','pipeline-status.json','latest-recovery.json']
print(json.dumps({n:json.loads((r/n).read_text()) for n in names if (r/n).exists()}))
'''
    return json.loads(ssh('python3 -', input=script))


def sync_receipts():
    DEST.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(['rsync', '-az', '--partial', '--delay-updates',
        '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15 -p ' + SSH_PORT,
        '--exclude=*.lock', '--exclude=__pycache__/',
        SSH_TARGET + ':' + REMOTE + '/', str(DEST) + '/'],
        text=True, capture_output=True, timeout=600)
    if result.returncode:
        raise RuntimeError('Receipt sync failed: ' + result.stderr[-500:])


def sync_completed_stage(state):
    pipeline = state.get('pipeline-status.json', {})
    if pipeline.get('phase') != 'awaiting_stage_backup':
        return
    stage = pipeline['stage']
    if not stage.replace('-', '').isalnum():
        raise RuntimeError('Unexpected stage identifier')
    completion_path = DEST / 'stages' / stage / 'complete.json'
    if not completion_path.exists():
        return
    raw = completion_path.read_bytes()
    completion = json.loads(raw)
    recovery = state['latest-recovery.json']
    if recovery['stage'] != stage:
        raise RuntimeError('Endpoint recovery belongs to another stage')
    if stage in ['A-bridge','repeat-A-bridge']:
        intermediate=stage.replace('bridge','cpt')
        prior_ack=ART/'stage-backups'/f'{intermediate}.json'
        intermediate_export=ART/'retained/models'/intermediate
        # Full prior CPT recovery remains available until this stage's full
        # recovery finishes replacing it. Keep the three final pilot exports.
        if prior_ack.exists() and intermediate_export.exists():
            prior=json.loads(prior_ack.read_text())
            if not prior.get('all_files_verified'):
                raise RuntimeError('Cannot retire an unverified intermediate export')
            shutil.rmtree(intermediate_export)
    files = {}
    for name, expected in completion['model_files'].items():
        files[name] = backup_stable(completion['model'] + '/' + name,
                    ART/'retained/models'/stage/name, expected)
    files['latest-recovery.pt'] = backup_stable(recovery['checkpoint'],
                     ART/'retained/latest-recovery.pt', recovery, rolling=True)
    marker = dict(all_files_verified=True, stage=stage, complete_sha256=hashlib.sha256(raw).hexdigest(),
                  recovery=recovery, files=files, utc=dt.datetime.now(dt.timezone.utc).isoformat())
    (ART/'stage-backups').mkdir(exist_ok=True)
    write(ART/'stage-backups'/f'{stage}.json', marker)
    command = 'python3 -'
    script = "import pathlib,json\np=pathlib.Path(" + repr(REMOTE+'/stage-backups/'+stage+'.json') + ")\np.parent.mkdir(exist_ok=True)\nt=p.with_suffix('.tmp')\nt.write_text(" + repr(json.dumps(marker)) + ")\nt.replace(p)\n"
    ssh(command, input=script)
    event('stage_backup_verified', stage=stage, files=len(files))


def backup_stable(path, destination, expected, rolling=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # During a rolling transfer the remote source is authoritative and must
    # remain immutable until this local SHA256 acknowledgement is written.
    # Existing historical backups are never overwritten.
    required = expected['bytes'] - (destination.stat().st_size if rolling and destination.exists() else 0)
    if shutil.disk_usage(destination.parent).free < max(0, required) + 1536 * 1024**2:
        raise RuntimeError('Insufficient local capacity; retain remote recovery and pause further training')
    result = subprocess.run(['rsync', '-az', '--partial'] + (['--inplace'] if rolling else []) + [
        '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15 -p ' + SSH_PORT,
        SSH_TARGET + ':' + path, str(destination)],
        text=True, capture_output=True, timeout=3600)
    if result.returncode:
        raise RuntimeError('Recovery/model transfer failed: ' + result.stderr[-500:])
    sha = sha_file(destination)
    if destination.stat().st_size != expected['bytes'] or sha != expected['sha256']:
        raise RuntimeError('Backup checksum mismatch: ' + str(destination))
    return dict(path=str(destination), bytes=destination.stat().st_size, sha256=sha)


def terminal_backup(state):
    # Child groups are stopped first so all checkpoint/model sources are stable.
    for name in owned_services():
        # Services that do not exist or already exited require no action.
        ssh('supervisorctl status ' + name + ' 2>/dev/null || true')
    sync_receipts()
    inventory = json.loads(ssh('python3 -', input='''import pathlib,hashlib,json
r=pathlib.Path('/dev/shm/video-model-validation48');files={}
latest=pathlib.Path('/workspace/video-model/validation-48h/latest-recovery.json')
active=json.loads(latest.read_text())['checkpoint'] if latest.exists() else None
if r.exists():
 for p in r.rglob('*'):
  if not p.is_file():continue
  name=str(p.relative_to(r))
  if name=='prior-prompts.sqlite' or name.startswith('models/') or name.startswith('candidates/') and name.endswith('.jsonl') or name.startswith('cpt-candidates/') and p.name in ['documents.jsonl','offsets.npy'] or str(p)==active:
   if '.next.' in name:continue
   with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
   files[name]={'bytes':p.stat().st_size,'sha256':sha}
print(json.dumps(files))
''', timeout=300))
    verified = {}
    for relative, expected in inventory.items():
        target = ART / 'retained' / ('latest-recovery.pt' if relative.endswith('.pt') else relative)
        verified[relative] = backup_stable(WORK + '/' + relative, target, expected, rolling=relative.endswith('.pt'))
    manifest = dict(all_files_verified=True, verified_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    files=verified, receipts=str(DEST), state=state,
                    original_backup_retained=True, policy='all completed exports plus latest full recovery')
    write(ART / 'backup-verified.json', manifest)
    return manifest


def owned_services():
    base = ['video-model-validation48', 'video-model-validation48-data', 'video-model-validation48-cpt',
            'video-model-validation48-suite', 'video-model-validation48-math', 'video-model-validation48-prepare']
    # A scoped registry lets new bounded CPU helpers join shutdown ownership
    # without replacing the running monitor for each preparation task.
    extra = json.loads(ssh('python3 -', input="import pathlib,json\np=pathlib.Path('/workspace/video-model/validation-48h/owned-services.json')\nprint(p.read_text() if p.exists() else '[]')\n"))
    for name in extra:
        if not isinstance(name, str) or not name.startswith('video-model-validation48-') or not name.replace('-', '').isalnum():
            raise ValueError('Unexpected new-phase service registry entry')
    return list(dict.fromkeys(base + extra))


def stop_owned_services():
    for service in owned_services():
        # Limit control to this new phase; do not touch original/comparator or
        # provider management services. SIGTERM allows a saved recovery.
        result = subprocess.run(SSH + ['supervisorctl stop ' + service], capture_output=True, text=True, timeout=720)
        if result.returncode and not any(s in result.stdout for s in ['no such process', 'not running']):
            raise RuntimeError('Could not stop owned stage: ' + service)


def stop_instance(manifest):
    if not manifest['all_files_verified']:
        raise RuntimeError('Backup prerequisite failed')
    # The existing scoped provider stop helper additionally checks the original
    # verified backup and GPU-process guard. Never print scoped API secrets.
    result = subprocess.run(SSH + [
        'source /opt/supervisor-scripts/utils/environment.sh; '
        '/venv/main/bin/python /workspace/video-model/stop_instance.py'],
        capture_output=True, text=True, timeout=120)
    if result.returncode == 0 and 'stop_request_accepted' in result.stdout:
        event('stop_request_accepted', actual_provider_stopped_confirmed=False)
        return True
    event('stop_unconfirmed', returncode=result.returncode, detail=result.stderr[-500:])
    return False


def main():
    if not SSH_TARGET or not all(c.isalnum() or c in '@._-' for c in SSH_TARGET):
        raise RuntimeError('Set VIDEO_MODEL_SSH_TARGET to the authorized SSH user@host')
    ART.mkdir(parents=True, exist_ok=True)
    lock = (ART / 'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    event('monitor_started', preparation_deadline_utc='2026-10-20T19:48:39+00:00',
          new_gpu_budget_started=False, old_monitor_restarted=False)
    while True:
        try:
            state = remote_state()
            budget = state.get('budget.json')
            pipeline = state.get('pipeline-status.json', {})
            if budget:
                if budget['max_gpu_seconds'] != 48*3600 or budget['deadline_unix'] - budget['started_unix'] != 48*3600:
                    raise RuntimeError('Invalid or extended GPU budget')
                deadline = budget['deadline_unix']
            else:
                deadline = PREP_END
            terminal = pipeline.get('status') in ['complete', 'failed', 'stopped']
            if time.time() >= deadline or terminal:
                event('terminal_or_deadline', deadline=deadline, pipeline=pipeline)
                stop_owned_services()
                state = remote_state()
                manifest = terminal_backup(state)
                if stop_instance(manifest):
                    return
            else:
                sync_receipts()
                sync_completed_stage(state)
                event('synced', preparation=state.get('preparation-status.json', {}).get('phase'),
                      pipeline=pipeline, budget=budget, training_ready=state.get('readiness.json', {}).get('ready', False))
        except Exception as exc:
            event('monitor_error', error=str(exc))
        time.sleep(120)


if __name__ == '__main__':
    main()
