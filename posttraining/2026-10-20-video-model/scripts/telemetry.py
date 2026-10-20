import datetime,json,subprocess,time
from common import ROOT,write_json

def main():
    fields=['timestamp','utilization.gpu','utilization.memory','memory.used','memory.total','power.draw','power.limit','temperature.gpu','clocks.sm','clocks.mem','pstate']
    end=datetime.datetime.fromisoformat('2026-10-20T19:48:39+00:00').timestamp()
    with open(ROOT/'gpu-telemetry.jsonl','a',buffering=1) as f:
        while time.time()<end:
            try:
                data=subprocess.check_output(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(', ')
                row=dict(zip(fields,data)); row['unix_time']=time.time()
                try:
                    p=json.loads((ROOT/'pipeline-status.json').read_text()); row['phase']=p.get('phase',p.get('status','unknown'))
                except Exception: row['phase']='setup'
                f.write(json.dumps(row)+'\n'); write_json(ROOT/'gpu-latest.json',row)
                if (ROOT/'final-report.json').exists(): break
            except Exception as e: f.write(json.dumps({'error':repr(e),'unix_time':time.time()})+'\n')
            time.sleep(5)

if __name__=='__main__': main()
