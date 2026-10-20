"""Regrade immutable completions after the documented code-execution bug fix."""
import collections
import datetime
import hashlib
import json
import shutil
from pathlib import Path

from common import ROOT, write_json
from evaluate import code_correct, extract_code
from audit_first_eval import execute


def main():
    source=ROOT/'evaluations'; legacy=source/'legacy-v1'; legacy.mkdir(exist_ok=True)
    report={'protocol':'v2-hashlib-entrypoint','generation_rerun':False,'reason':'A canonical HumanEval MD5 solution fails because hashlib was not preloaded in the sealed worker; choose the requested function deterministically rather than the longest code block.','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':{}}
    for name,base in [('base-dev',True),('sft-2000',False)]:
        old=source/(name+'.json'); sample=source/(name+'.samples.jsonl')
        for path in [old,sample]:
            dest=legacy/path.name
            if not dest.exists(): shutil.copyfile(path,dest)
        summary=json.loads((legacy/old.name).read_text())
        rows=[json.loads(s) for s in (legacy/sample.name).read_text().splitlines()]
        changes=[]
        for row in rows:
            if row['task']!='humaneval': continue
            correct=code_correct(extract_code(row['completion'],row['prompt'],base,row['entry_point']),row['test'],row['entry_point'])
            if correct!=row['correct']: changes.append({'id':row['id'],'old':row['correct'],'new':correct})
            row['correct']=correct
        metrics={}
        for task in ['gsm8k','math500','humaneval']:
            selected=[r for r in rows if r['task']==task]; count=sum(r['correct'] for r in selected)
            metrics[task]={'correct':count,'n':len(selected),'accuracy':count/len(selected)}
        summary.update(metrics=metrics,macro_accuracy=sum(m['accuracy'] for m in metrics.values())/3,evaluation_protocol=report['protocol'],code_extraction='first complete code fence defining requested entrypoint; prompt context retained; no test-guided selection',sandbox_revision='v2: hashlib/md5 preloaded before chroot; same isolation and resource limits',rescored_without_new_generation=True,legacy_result=str(legacy/old.name))
        temp=sample.with_suffix('.jsonl.tmp'); temp.write_text(''.join(json.dumps(r)+'\n' for r in rows)); temp.replace(sample)
        write_json(old,summary)
        report['results'][name]={'metrics':metrics,'macro_accuracy':summary['macro_accuracy'],'changed_items':changes,'legacy_sha256':{p.name:hashlib.sha256((legacy/p.name).read_bytes()).hexdigest() for p in [old,sample]}}
    # Confirm the repaired sandbox against all 32 development references.
    from datasets import load_dataset
    revision=json.loads((ROOT/'benchmark-provenance.json').read_text())['openai/openai_humaneval']
    dev_ids={r['id'] for r in rows if r['task']=='humaneval'}
    refs=[r for r in load_dataset('openai/openai_humaneval',revision=revision,split='test') if r['task_id'] in dev_ids]
    tests=[{'id':r['task_id'],**execute(r['prompt']+r['canonical_solution'],r)} for r in refs]
    assert len(tests)==32 and all(r['passed'] for r in tests),tests
    report['reference_oracle']={'n':32,'passed':32,'items':tests}
    # A wrong digest is still rejected; the resource/syscall policy was not relaxed.
    assert not code_correct('def f(x): return "wrong"','def check(f): assert f("x")=="9dd4e461268c8034f5c8564e155c67a6"','f')
    text='```python\ndef wanted(x): return x+1\n```\n```python\ndef unrelated(x):\n    """A much longer unrelated function."""\n    return x-100\n```'
    assert code_correct(extract_code(text,'def wanted(x):\n    """Add one."""\n',False,'wanted'),'def check(f): assert f(2)==3','wanted')
    best_path=ROOT/'best-model.json'; best=json.loads(best_path.read_text())
    assert best['stage']=='sft' and best['step']==2000,'Only migrate the known selected checkpoint'
    if not (ROOT/'best-model-legacy-v1.json').exists(): shutil.copyfile(best_path,ROOT/'best-model-legacy-v1.json')
    best['dev_macro_accuracy']=report['results']['sft-2000']['macro_accuracy']; best['evaluation_protocol']=report['protocol']
    write_json(best_path,best)
    write_json(ROOT/'evaluation-amendment-v2.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
