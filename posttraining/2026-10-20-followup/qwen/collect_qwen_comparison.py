"""Copy, verify and compare completed comparator outputs, then release stop guard."""
import base64
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from compare_final_evaluation import EXPECTED, PROTOCOL, paired_bootstrap, wilson

LOCAL=Path(__file__).resolve().parents[1]
ART=LOCAL/'artifacts'
DEST=ART/'qwen-eval-inputs'
SSH_TARGET=os.environ.get('VIDEO_MODEL_SSH_TARGET','')
SSH_PORT=str(int(os.environ.get('VIDEO_MODEL_SSH_PORT','22')))
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p',SSH_PORT,SSH_TARGET,'python3 -']
REMOTE='/workspace/video-model/comparisons/qwen-final-20261020'


def ssh(script):
    if not SSH_TARGET or not all(c.isalnum() or c in '@._-' for c in SSH_TARGET):
        raise RuntimeError('Set VIDEO_MODEL_SSH_TARGET to the authorized SSH user@host')
    p=subprocess.run(SSH,input=script,text=True,capture_output=True,timeout=180)
    if p.returncode:raise RuntimeError(p.stderr[-1200:])
    return json.loads(p.stdout)


def main():
    payload=ssh('''import pathlib,json,base64,hashlib
r=pathlib.Path(REMOTE)
p=r/'complete.json'
if not p.exists():
 print(json.dumps({'ready':False,'status':json.loads((r/'status.json').read_text())}));raise SystemExit(0)
raw=p.read_bytes();manifest=json.loads(raw);files={}
assert manifest['status']=='complete'
for name,expected in manifest['files'].items():
 path=r/name
 assert path.is_file() and path.parent==r
 data=path.read_bytes()
 if name=='controller.log':
  # Supervisor appends status after the completion snapshot. Preserve and
  # verify exactly the immutable prefix declared in that snapshot.
  assert len(data)>=expected['bytes'];data=data[:expected['bytes']]
 assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'],name
 files[name]=base64.b64encode(data).decode()
print(json.dumps({'ready':True,'manifest':base64.b64encode(raw).decode(),'files':files}))
'''.replace('REMOTE',repr(REMOTE)))
    if not payload['ready']:
        print(json.dumps(payload,ensure_ascii=False,indent=2));return
    DEST.mkdir(exist_ok=True)
    manifest_raw=base64.b64decode(payload['manifest'])
    manifest=json.loads(manifest_raw)
    for name,encoded in payload['files'].items():
        data=base64.b64decode(encoded);expected=manifest['files'][name]
        assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256']
        p=DEST/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(data);tmp.replace(p)
    (DEST/'complete.json').write_bytes(manifest_raw)
    plan=json.loads((DEST/'plan.json').read_text())
    for name,digest in plan['frozen_sources'].items():
        assert hashlib.sha256((ART/'final-eval-inputs'/name).read_bytes()).hexdigest()==digest
    summaries={};indices={};hashes={}
    paths={'base':ART/'final-eval-inputs/evaluations/base-final',
           'posttrained':ART/'final-eval-inputs/evaluations/best-final',
           'qwen-0.5b':DEST/'qwen-0.5b','qwen-1.5b':DEST/'qwen-1.5b'}
    for label,stem in paths.items():
        jp=Path(str(stem)+'.json');sp=Path(str(stem)+'.samples.jsonl')
        summary=json.loads(jp.read_text());rows=[json.loads(x) for x in sp.read_text().splitlines()]
        assert Counter(x['task'] for x in rows)==EXPECTED
        assert len(rows)==1791 and len({(x['task'],x['id']) for x in rows})==1791
        assert len({(x['task'],x['prompt']) for x in rows})==1791
        for k,v in PROTOCOL.items():
            if label.startswith('qwen') and k=='eos_token_ids':continue
            assert summary[k]==v,(label,k)
        if label.startswith('qwen'):
            model=next(x for x in plan['models'] if x['key']==label)
            assert summary['model']==model['repo'] and summary['model_revision']==model['revision']
            assert summary['eos_token_strings']==['<|im_end|>','<|endoftext|>']
            assert len(set(summary['eos_token_ids']))==2 and all(isinstance(i,int) and i>32001 for i in summary['eos_token_ids'])
            assert summary['base_prompting'] is False
            assert summary['benchmark_sha256']==plan['benchmark_sha256']
            assert summary['frozen_sources']==plan['frozen_sources']
        assert all(type(r['correct']) is bool and r['split']=='final' for r in rows)
        indices[label]={(r['task'],r['id']):r for r in rows}
        summaries[label]={}
        for task,n in EXPECTED.items():
            k=sum(r['correct'] for r in rows if r['task']==task);m=summary['metrics'][task]
            assert (k,n)==(m['correct'],m['n']) and abs(m['accuracy']-k/n)<1e-12
            summaries[label][task]={**m,'wilson_95_ci':wilson(k,n)}
        for p in [jp,sp]:hashes[str(p.relative_to(ART))]=hashlib.sha256(p.read_bytes()).hexdigest()
    ref=indices['posttrained']
    for label,index in indices.items():
        assert index.keys()==ref.keys()
        for key in ref:
            content=lambda r:{k:v for k,v in r.items() if k not in ['completion','correct']}
            assert content(ref[key])==content(index[key]),(label,key)
    deltas={}
    for j,label in enumerate(['qwen-0.5b','qwen-1.5b']):
        deltas[label]={}
        for i,(task,n) in enumerate(EXPECTED.items()):
            keys=[k for k in ref if k[0]==task]
            gains=sum(not ref[k]['correct'] and indices[label][k]['correct'] for k in keys)
            losses=sum(ref[k]['correct'] and not indices[label][k]['correct'] for k in keys)
            deltas[label][task]={'qwen_minus_posttrained_pp':100*(gains-losses)/n,
                                 'qwen_gained_items':gains,'qwen_lost_items':losses,
                                 'paired_bootstrap_95_ci_pp':paired_bootstrap(gains,losses,n,20261029+10*j+i)}
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    result={'status':'verified_complete','verified_utc':now,'finished_utc':manifest['finished_utc'],
            'metrics':summaries,'qwen_minus_posttrained':deltas,'plan':plan,'input_sha256':hashes,
            'comparison_manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),
            'verification':{'1791_unique_rows_each':True,'benchmark_content_identical':True,'correct_counts_reaggregated':True,
                            'frozen_generator_and_grader_source_hashes_match':True,'all_comparison_snapshot_files_verified':True,
                            'new_inference':False,'new_grading':False},
            'limitations':['Native tokenizer/chat templates and EOS IDs differ; same user instructions and semantic stop tokens.',
                           'Qwen native templates may supply a default system message; rendered prompts are not identical.',
                           '2048-token total cap and 1100-token prompt cap apply per model tokenizer, so text truncation/generation budgets can differ.',
                           'One greedy generation per held-out question; question sampling intervals do not cover training-seed variation.',
                           'No subsequent tuning or selection of the L20 checkpoint from comparator results.']}
    (ART/'qwen-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    names={'base':'原始 L20 基座','posttrained':'L20 后训练 SFT12000','qwen-0.5b':'Qwen2.5-0.5B-Instruct','qwen-1.5b':'Qwen2.5-1.5B-Instruct'}
    lines=['# Qwen 与 L20 同题同判分对照','',f'核验时间 UTC：{now}。每个模型1791题。','',
           '| 模型 | GSM8K（/1223） | MATH500（/436） | HumanEval（/132） |','|---|---:|---:|---:|']
    for label in paths:
        vals=[f"{summaries[label][t]['accuracy']*100:.2f}%（{summaries[label][t]['correct']}/{summaries[label][t]['n']}）" for t in EXPECTED]
        lines.append('| '+names[label]+' | '+' | '.join(vals)+' |')
    lines+=['','相对 L20 后训练模型的差距（Qwen减L20；正值代表Qwen更高）：','',
            '| 对照 | GSM8K | MATH500 | HumanEval |','|---|---:|---:|---:|']
    for label in deltas:lines.append('| '+names[label]+' | '+' | '.join(f"{deltas[label][t]['qwen_minus_posttrained_pp']:+.2f} 个百分点" for t in EXPECTED)+' |')
    lines+=['','题目ID、题目、答案和代码测试完全一致，逐题正确数已重算，原生成函数、math-verify和隔离代码执行器保持冻结。seed=20261019，BF16，greedy，batch32，总上下文2048、提示上限1100，最多生成1800且受剩余上下文限制。',
            '', '接口适配：Qwen使用各自原生分词器及聊天模板（可能加入默认system消息），结束符映射到原生im_end和endoftext，不能照搬L20的数字token ID。用户指令相同，渲染文本、实际可容纳文本和结束符数字ID不同。Qwen采样/重复惩罚默认值已清除，使用与L20一致的greedy设置。',
            '', '模型revision、完整哈希和配对差值95%区间见同名JSON。区间来自10000次固定种子的题目配对自助抽样；不是训练种子不确定性。这是原有冻结独立部分的对照，不是全官方benchmark分数，也没有用结果继续调模型。',
            '', '控制器日志备份为完成清单声明的不可变前缀；其后追加的备份等待状态不影响判分输出。']
    (ART/'qwen-comparison.md').write_text('\n'.join(lines)+'\n')
    ack={'all_files_verified':True,'comparison_manifest_sha256':result['comparison_manifest_sha256'],
         'verified_utc':now,'local_destination':str(DEST),'file_count':len(manifest['files']),
         'controller_log_is_declared_immutable_prefix':True}
    (ART/'qwen-backup-verified.json').write_text(json.dumps(ack,indent=2)+'\n')
    response=ssh('''import pathlib,json
r=pathlib.Path(REMOTE)
data=json.loads(ACK)
p=r/'local-backup-verified.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
print(json.dumps({'acknowledged':True}))
'''.replace('REMOTE',repr(REMOTE)).replace('ACK',repr(json.dumps(ack))))
    assert response['acknowledged']
    print(json.dumps({'status':result['status'],'metrics':summaries,'deltas':deltas,'acknowledged':True},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
