"""Only called after the user's authorized, verified off-box backup."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/workspace/posttrain-l20')

def check_validation_data_backup():
    phase=ROOT/'validation-48h'
    if not (phase/'owned-services.json').exists():
        return
    manifest_path=phase/'data-backup/manifest.json'
    acknowledgement=json.loads((phase/'data-backup/local-verified.json').read_text())
    manifest=json.loads(manifest_path.read_text())
    if (not acknowledgement.get('all_files_verified') or
        not acknowledgement.get('decompressed_original_bytes_verified') or
        acknowledgement['manifest_sha256']!=hashlib.sha256(manifest_path.read_bytes()).hexdigest()):
        raise RuntimeError('New phase data archives have not been verified off-box')
    expected={k:v['original'] for k,v in manifest['files'].items()}
    received={k:v['original'] for k,v in acknowledgement['files'].items()
              if v.get('original_bytes_verified')}
    if expected!=received:
        raise RuntimeError('New phase data backup inventory differs from acknowledgement')
    sources={Path(v['source']).name for v in manifest['files'].values()}
    if 'prior-code-families.sqlite' not in sources:
        raise RuntimeError('Prior code exclusion index missing from backup')
    function_index=phase/'prior-code-functions-complete.json'
    if function_index.exists() or (phase/'budget.json').exists():
        identity=json.loads(function_index.read_text())
        matches=[v for v in manifest['files'].values() if v['source']==identity['path']]
        if (len(matches)!=1 or matches[0]['original']['sha256']!=identity['sha256'] or
            matches[0]['original']['bytes']!=Path(identity['path']).stat().st_size):
            raise RuntimeError('Embedded-function exclusion index missing from verified backup')
    if (phase/'budget.json').exists():
        roles={v['role'] for v in manifest['files'].values()}
        if not {'packed-cpt-objective','packed-sft-objective'}<=roles:
            raise RuntimeError('Native packed objectives missing from new phase backup')
        launch=json.loads((phase/'launch.json').read_text())
        archived={v['source']:v for v in manifest['files'].values()}
        for stage in ['B1','A-cpt']:
            data=Path(launch['stages'][stage]['data'])
            native_path=data/'manifest.json'
            native=json.loads(native_path.read_text())
            for name in ['manifest.json','train.ids.u16','train.mask.u8']:
                path=str(data/name)
                expected_native=(dict(bytes=native_path.stat().st_size,
                    sha256=hashlib.sha256(native_path.read_bytes()).hexdigest())
                    if name=='manifest.json' else native['files'][name])
                if path not in archived or archived[path]['original']!=expected_native:
                    raise RuntimeError('Incomplete native objective backup: '+path)

def main():
    marker=json.loads((ROOT/'backup-verified.json').read_text())
    manifest_hash=hashlib.sha256((ROOT/'backup-manifest.json').read_bytes()).hexdigest()
    if not marker.get('all_files_verified') or marker['backup_manifest_sha256']!=manifest_hash: raise RuntimeError('Backup has not been verified')
    status=json.loads((ROOT/'pipeline-status.json').read_text())
    if status.get('status') not in ['complete','failed','stopped']: raise RuntimeError('Training is still active')
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if gpu: raise RuntimeError('GPU processes are still active; refusing instance stop')
    check_validation_data_backup()
    instance=os.environ.get('CONTAINER_ID'); key=os.environ.get('CONTAINER_API_KEY')
    if not instance or not key: raise RuntimeError('Vast instance-scoped credentials unavailable')
    (ROOT/'stop-requested.json').write_text(json.dumps({'time':time.time(),'instance':instance,'backup_manifest_sha256':manifest_hash}))
    result=subprocess.run(['vastai','stop','instance',instance,'--api-key',key],text=True,capture_output=True,timeout=45)
    if result.returncode: raise RuntimeError('Vast stop request failed, exit '+str(result.returncode))
    print(json.dumps({'stop_request_accepted':True,'instance':instance,'time':time.time()}),flush=True)

if __name__=='__main__': main()
