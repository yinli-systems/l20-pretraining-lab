import hashlib,json
from common import ROOT,write_json

def main():
    from summarize import write_report
    write_report()
    files=[]
    for folder in ['models/best','models/sft-selected']:
        p=ROOT/folder
        if p.exists(): files.extend(x for x in p.rglob('*') if x.is_file())
    if (ROOT/'resume.pt').exists() and (ROOT/'resume-ready.json').exists(): files.append(ROOT/'resume.pt')
    # On an early failure, retain any completed pilot checkpoint too.
    if not any('models/best/' in str(x) for x in files):
        for p in (ROOT/'models').glob('pilot-*'):
            files.extend(x for x in p.rglob('*') if x.is_file())
    for pattern in ['*.py','*.sh','*.conf','*.json','*.md','*.tar.gz','requirements.lock']:
        files.extend(p for p in ROOT.glob(pattern) if p.is_file() and p.name not in ['backup-manifest.json','backup-verified.json','gpu-latest.json','pipeline-status.json'])
    manifest={}
    for p in sorted(set(files)):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
        manifest[str(p.relative_to(ROOT))]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
    write_json(ROOT/'backup-manifest.json',{'files':manifest,'includes_resume': 'resume.pt' in manifest})

if __name__=='__main__': main()
