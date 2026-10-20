"""Only called after the user's authorized, verified off-box backup."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/workspace/posttrain-l20')

def main():
    marker=json.loads((ROOT/'backup-verified.json').read_text())
    manifest_hash=hashlib.sha256((ROOT/'backup-manifest.json').read_bytes()).hexdigest()
    if not marker.get('all_files_verified') or marker['backup_manifest_sha256']!=manifest_hash: raise RuntimeError('Backup has not been verified')
    status=json.loads((ROOT/'pipeline-status.json').read_text())
    if status.get('status') not in ['complete','failed','stopped']: raise RuntimeError('Training is still active')
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if gpu: raise RuntimeError('GPU processes are still active; refusing instance stop')
    instance=os.environ.get('CONTAINER_ID'); key=os.environ.get('CONTAINER_API_KEY')
    if not instance or not key: raise RuntimeError('Vast instance-scoped credentials unavailable')
    (ROOT/'stop-requested.json').write_text(json.dumps({'time':time.time(),'instance':instance,'backup_manifest_sha256':manifest_hash}))
    result=subprocess.run(['vastai','stop','instance',instance,'--api-key',key],text=True,capture_output=True,timeout=45)
    if result.returncode: raise RuntimeError('Vast stop request failed, exit '+str(result.returncode))
    print(json.dumps({'stop_request_accepted':True,'instance':instance,'time':time.time()}),flush=True)

if __name__=='__main__': main()
