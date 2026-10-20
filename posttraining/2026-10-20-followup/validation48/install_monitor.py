"""Install exactly one new phase monitor; leave historical service untouched."""
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / 'Library/Application Support/Kevin/video-model-validation48-20261020'
LABEL = 'com.kevin.video-model-validation48-20261020'


def main():
    target = os.environ.get('VIDEO_MODEL_SSH_TARGET', '')
    port = str(int(os.environ.get('VIDEO_MODEL_SSH_PORT', '22')))
    if not target or not all(c.isalnum() or c in '@._-' for c in target):
        raise RuntimeError('Set VIDEO_MODEL_SSH_TARGET before installing a monitor')
    runtime_art = RUNTIME/'artifacts/validation48'
    runtime_art.mkdir(parents=True, exist_ok=True)
    (RUNTIME/'scripts').mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'validation48/monitor.py', RUNTIME/'scripts/monitor.py')
    art = ROOT/'artifacts/validation48'
    art.mkdir(parents=True, exist_ok=True)
    for name in ['remote-receipts','retained','stage-backups']:
        (runtime_art/name).mkdir(exist_ok=True)
    for name in ['remote-receipts','retained','stage-backups','monitor-status.json','monitor.log','backup-verified.json']:
        p = art/name
        target = runtime_art/name
        if p.is_symlink():
            if p.resolve() != target:
                raise RuntimeError('Existing link points to another monitor: '+name)
        elif p.exists():
            raise RuntimeError('Refuse to replace existing monitor artifact: '+name)
        else:
            p.symlink_to(target)
    plist = Path.home()/'Library/LaunchAgents'/f'{LABEL}.plist'
    config = dict(Label=LABEL, ProgramArguments=['/usr/bin/caffeinate','-i','/usr/bin/python3',str(RUNTIME/'scripts/monitor.py')],
                  WorkingDirectory=str(RUNTIME), RunAtLoad=True,
                  EnvironmentVariables=dict(VIDEO_MODEL_SSH_TARGET=target, VIDEO_MODEL_SSH_PORT=port),
                  KeepAlive=dict(SuccessfulExit=False), ThrottleInterval=60,
                  StandardOutPath=str(runtime_art/'monitor.log'),
                  StandardErrorPath=str(runtime_art/'monitor.log'))
    if plist.exists():
        raise RuntimeError('Monitor already installed; inspect status instead of creating another')
    plist.write_bytes(plistlib.dumps(config))
    subprocess.run(['launchctl','bootstrap',f'gui/{os.getuid()}',str(plist)],check=True)
    receipt = dict(label=LABEL, runtime=str(RUNTIME), plist=str(plist),
                   old_monitor_restarted=False, local_status=str(art/'monitor-status.json'))
    (art/'monitor-installation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
