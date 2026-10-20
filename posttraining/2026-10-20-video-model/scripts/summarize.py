"""Create a human-readable final result with paired, held-out uncertainty."""
import json,math
import numpy as np
from common import ROOT,SEED,write_json

def wilson(k,n):
    if not n: return [None,None]
    z=1.959963984540054; p=k/n; d=1+z*z/n
    center=(p+z*z/(2*n))/d
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,center-radius),min(1,center+radius)]

def write_report():
    naming=ROOT/'project-display-name.json'
    display_name=json.loads(naming.read_text()).get('display_name','L20 English Math and Code Reasoning') if naming.exists() else 'L20 English Math and Code Reasoning'
    final=ROOT/'final-report.json'
    if not final.exists():
        status=json.loads((ROOT/'pipeline-status.json').read_text())
        (ROOT/'run-status.md').write_text('# 运行状态\n\n'+str(status.get('status'))+'\n\n'+str(status.get('error',''))+'\n\n可用模型和恢复文件会随备份一并保留。此状态不代表后训练与最终评测已经完成。\n')
        return
    r=json.loads(final.read_text()); before=r['base']; after=r['posttrained']
    paths=[ROOT/'evaluations/base-final.samples.jsonl',ROOT/'evaluations/best-final.samples.jsonl']
    comparison={}; macro_samples=[]; rng=np.random.default_rng(SEED)
    if all(p.exists() for p in paths):
        datasets=[{(x['task'],x['id']):x for x in (json.loads(s) for s in p.read_text().splitlines())} for p in paths]
        if set(datasets[0])!=set(datasets[1]): raise RuntimeError('Final evaluation item mismatch')
        for task in sorted(after['metrics']):
            keys=sorted(k for k in datasets[0] if k[0]==task)
            delta=np.asarray([int(datasets[1][k]['correct'])-int(datasets[0][k]['correct']) for k in keys],dtype=np.float64)
            draws=[]
            for _ in range(10): draws.append(delta[rng.integers(0,len(delta),(1000,len(delta)))].mean(1))
            draws=np.concatenate(draws); macro_samples.append(draws)
            comparison[task]={'paired_delta':float(delta.mean()),'paired_bootstrap_95_ci':np.quantile(draws,[.025,.975]).tolist(),'n':len(keys)}
        macro=np.stack(macro_samples).mean(0)
        comparison['macro']={'paired_delta':after['macro_accuracy']-before['macro_accuracy'],'paired_bootstrap_95_ci':np.quantile(macro,[.025,.975]).tolist(),'method':'10,000 paired, task-stratified bootstrap draws; fixed seed; item-sampling uncertainty only'}
    write_json(ROOT/'paired-comparison.json',comparison)
    rows=[f'# {display_name} 训练结果','','最终状态：训练和最终隔离集评测已完成。','','最佳模型：`models/best/`。独立的 SFT 参考模型为 `models/sft-selected/`；恢复状态为 `resume.pt`。','','| 最终隔离测试 | 基座 | 后训练 | 后训练 Wilson 95% 区间 | 提升 |','|---|---:|---:|---:|---:|']
    for task,b in after['metrics'].items():
        a=before['metrics'][task]; lo,hi=wilson(b['correct'],b['n']); diff=b['accuracy']-a['accuracy']
        rows.append(f"| {task} (n={b['n']}) | {a['accuracy']:.2%} | {b['accuracy']:.2%} | {lo:.2%}–{hi:.2%} | {diff*100:+.2f} pp |")
    rows+=['',f"三项等权平均：{before['macro_accuracy']:.2%} → {after['macro_accuracy']:.2%}。",'', '这些分数来自预先隔离的测试部分，不是完整官方 benchmark 的成绩。开发集只用于选择 checkpoint，最终隔离集没有参与训练或选模型。','', '评测：greedy，原生 2048 上下文，输入上限 1100 tokens（从左截断），最多生成 1800 tokens（同时受 2048 总上下文限制）；数学用 math-verify，代码实际执行测试。区间只反映题目抽样不确定性，不包含训练种子差异，也不能排除未检测出的语义近重复。','', '训练与硬件的逐步记录见 `logs/*-metrics.jsonl`、`gpu-telemetry.jsonl`、`data-manifest.json` 和 `phase-history.jsonl`。配置比较的预检吞吐与全程平均分开记录。','', '本报告在备份校验和停机之前生成；它本身不证明备份成功或实例已经停机。备份与停止请求的确认见本机 `local-monitor-status.json`。','']
    (ROOT/'final-report.md').write_text('\n'.join(rows))
    model=ROOT/'models/best'
    card=['---','base_model: AliceYin/L20-1B-20B-Base','language: en','pipeline_tag: text-generation','tags:','- math','- code','- reasoning','- instruction-tuned','---','',f'# {display_name}','','Full-parameter post-training of [AliceYin/L20-1B-20B-Base](https://huggingface.co/AliceYin/L20-1B-20B-Base), using the original 2048-token context window. The tokenizer adds ChatML boundary tokens; inference configuration suppresses unused vocabulary padding rows.','','Training uses BF16 computation with FP32 master weights and AdamW state. The run compares learning rates, selects checkpoints on a separate development partition, then optionally applies rejection-sampling fine-tuning with mathematical answer checks or execution tests. This is not PPO or GRPO.','','Training sources and attribution:','','- [NVIDIA OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)','- [NVIDIA OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)','- [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)','- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)','- [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k), when enrichment completes','- [NVIDIA OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning), when enrichment completes','','Pinned revisions, filters and exact counts are recorded in the accompanying run receipts. R1 math uses supplied completion/correctness flags; R1 code traces are checked for completeness and Python syntax, and are not independently execution-verified as a whole dataset. RFT code outputs are execution-tested.','','## Held-out results','','These are the reserved portions of each benchmark, not full official benchmark scores. Generation is greedy, with at most 1800 new tokens (further capped by the native context) and a 1100-token prompt cap.','','| Task | Items | Base | Post-trained |','|---|---:|---:|---:|']
    for task,m in after['metrics'].items(): card.append(f"| {task} | {m['n']} | {before['metrics'][task]['accuracy']:.2%} | {m['accuracy']:.2%} |")
    card+=['','## Usage','','```python','import torch','from transformers import AutoTokenizer, AutoModelForCausalLM','','path = "/path/to/models/best"','tokenizer = AutoTokenizer.from_pretrained(path)','model = AutoModelForCausalLM.from_pretrained(','    path, torch_dtype=torch.bfloat16, device_map="auto",','    attn_implementation="sdpa",',')','messages = [{"role": "user", "content": "Solve 3x + 7 = 22 step by step."}]','inputs = tokenizer.apply_chat_template(','    messages, add_generation_prompt=True, tokenize=True,','    return_dict=True, return_tensors="pt",',').to(model.device)','with torch.inference_mode():','    output = model.generate(**inputs, max_new_tokens=1800, do_sample=False)','print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))','```','','The run does not establish global optimality or advanced contest-level reasoning. It can still make mathematical, coding and factual errors. Responses longer than the native context are not supported. No public Hub release was performed by the training workflow.','']
    if model.exists(): (model/'README.md').write_text('\n'.join(card))

if __name__=='__main__': write_report()
