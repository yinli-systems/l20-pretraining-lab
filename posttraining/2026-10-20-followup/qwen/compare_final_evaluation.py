"""Audit completed final evaluation receipts without generating or regrading answers."""
import argparse
import datetime as dt
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


EXPECTED = {'gsm8k': 1223, 'math500': 436, 'humaneval': 132}
PROTOCOL = {
    'split': 'final', 'seed': 20261019, 'context': 2048,
    'max_new_tokens': 1800,
    'effective_max_new_tokens': 'min(1800, 2048 - padded_prompt_length)',
    'prompt_token_cap': 1100, 'prompt_truncation': 'left',
    'generation': 'greedy', 'eos_token_ids': [32001, 0],
    'evaluation_protocol': 'v2-hashlib-entrypoint',
    'math_grader': 'math-verify',
    'code_execution': 'empty chroot, unprivileged UID, seccomp syscall allowlist, 3s CPU / 6s wall / 768MiB address space',
}


def read_json(path):
    return json.loads(path.read_text())


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [max(0, center-half), min(1, center+half)]


def paired_bootstrap(gains, losses, n, seed):
    rng = random.Random(seed)
    draws = []
    for _ in range(10000):
        delta = 0
        for _ in range(n):
            j = rng.randrange(n)
            delta += 1 if j < gains else (-1 if j < gains + losses else 0)
        draws.append(100 * delta / n)
    draws.sort()
    return [draws[249], draws[9749]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--artifacts', type=Path, required=True)
    a = p.parse_args()
    final = read_json(a.inputs/'final-report.json')
    assert final['status'] == 'complete', 'Final report is not complete'
    selection = read_json(a.inputs/'best-model.json')
    assert selection == final['selection']
    assert selection['stage'] == 'sft' and selection['step'] == 12000
    remote = read_json(a.inputs/'remote-hashes.json')
    inputs = {}
    for name, expected in remote['files'].items():
        path = a.inputs/name
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == expected['sha256'], name + ' hash mismatch'
        assert len(raw) == expected['bytes'], name + ' size mismatch'
        inputs[name] = {'path': str(path.resolve()), 'sha256': digest,
                        'bytes': len(raw), 'remote_hash_verified': True}
    indices = {}
    metrics = {}
    reports = {}
    for name, final_key, base in [('base-final', 'base', True), ('best-final', 'posttrained', False)]:
        report = read_json(a.inputs/'evaluations'/(name+'.json'))
        assert report == final[final_key], name + ' differs from final report'
        assert report['base_prompting'] is base
        for key, value in PROTOCOL.items():
            assert report[key] == value, (name, key, report.get(key))
        rows = [json.loads(line) for line in (a.inputs/'evaluations'/(name+'.samples.jsonl')).read_text().splitlines()]
        assert len(rows) == sum(EXPECTED.values())
        assert Counter(x['task'] for x in rows) == EXPECTED
        index = {}
        prompts = set()
        for row in rows:
            key = (row['task'], row['id'])
            assert key not in index, (name, 'duplicate id', key)
            assert row['split'] == 'final' and type(row['correct']) is bool
            assert isinstance(row['completion'], str)
            prompt_key = (row['task'], row['prompt'])
            assert prompt_key not in prompts, (name, 'duplicate prompt', key)
            prompts.add(prompt_key)
            index[key] = row
        indices[name] = index
        metrics[name] = {}
        for task, n in EXPECTED.items():
            correct = sum(x['correct'] for x in rows if x['task'] == task)
            m = report['metrics'][task]
            assert (correct, n) == (m['correct'], m['n'])
            assert abs(m['accuracy']-correct/n) < 1e-12
            metrics[name][task] = {**m, 'wilson_95_ci': wilson(correct, n)}
        macro = sum(x['accuracy'] for x in report['metrics'].values())/3
        assert abs(macro-report['macro_accuracy']) < 1e-12
        reports[name] = report
    left, right = indices['base-final'], indices['best-final']
    assert left.keys() == right.keys(), 'Question IDs differ'
    for key in left:
        content = lambda row: {k:v for k,v in row.items() if k not in ['completion','correct']}
        assert content(left[key]) == content(right[key]), ('Benchmark content differs', key)
    dev = a.artifacts/'rft-2-eval-inputs/base-dev.samples.jsonl'
    dev_rows = [json.loads(s) for s in dev.read_text().splitlines()]
    assert not set(left).intersection((x['task'], x['id']) for x in dev_rows)
    assert not {(x['task'],x['prompt']) for x in left.values()}.intersection((x['task'],x['prompt']) for x in dev_rows)
    for key in ['code_extraction', 'sandbox_revision']:
        assert reports['base-final'][key] == reports['best-final'][key]
    deltas = {}
    for i, (task, n) in enumerate(EXPECTED.items()):
        keys = [k for k in left if k[0] == task]
        gained = sum(not left[k]['correct'] and right[k]['correct'] for k in keys)
        lost = sum(left[k]['correct'] and not right[k]['correct'] for k in keys)
        b, c = metrics['base-final'][task], metrics['best-final'][task]
        assert gained-lost == c['correct']-b['correct']
        deltas[task] = {
            'delta_percentage_points': 100*(c['accuracy']-b['accuracy']),
            'gained_items': gained, 'lost_items': lost,
            'paired_bootstrap_95_ci_pp': paired_bootstrap(gained,lost,n,20261019+i),
        }
    now = dt.datetime.now(dt.timezone.utc)
    result = {
        'status': 'verified_complete', 'verified_utc': now.isoformat(),
        'verified_dubai': now.astimezone(dt.timezone(dt.timedelta(hours=4))).isoformat(),
        'pipeline_finished_utc': final['finished_utc'],
        'selected_checkpoint': selection, 'metrics': metrics, 'deltas': deltas,
        'macro_accuracy': {k: reports[k]['macro_accuracy'] for k in reports},
        'protocol': PROTOCOL,
        'prompt_adaptation': 'Base continuation; posttrained ChatML. Benchmark content identical; rendered prompts differ.',
        'verification': {
            '1791_rows_each': True, 'same_unique_task_ids_and_benchmark_content': True,
            'disjoint_from_development_ids_and_prompts': True,
            'counts_recomputed_from_all_verdicts': True, 'all_input_hashes_match_remote': True,
            'new_generation': False, 'new_grading': False,
            'failures_or_inconsistencies': [],
        },
        'inputs': inputs,
        'uncertainty': 'Per-score Wilson 95% intervals; paired bootstrap percentile 95% intervals with 10000 resamples, seeds 20261019..20261021. They describe finite benchmark uncertainty, not training-run variation or a causal decomposition.',
        'limitations': ['Single trained checkpoint and one frozen held-out split; no checkpoint revisions after observing final results.',
                       'Native 2048-token context and greedy decoding; not directly comparable to leaderboards with other protocols.',
                       'Math answer grading does not certify reasoning steps; code pass rate covers supplied HumanEval tests only.',
                       'Original grading verdicts were aggregated, not independently regraded in this comparison.'],
    }
    out = a.artifacts/'final-eval-comparison.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    titles = {'gsm8k':'GSM8K 数学','math500':'MATH500 数学','humaneval':'HumanEval 代码执行通过率'}
    lines = ['# 完整独立评测：原模型与后训练模型', '',
             f"流水线完成 UTC：{final['finished_utc']}。核验时间（迪拜）：{result['verified_dubai']}。", '',
             '最终模型是开发集预先选定的 SFT 12000 步；两轮 RFT 均未超过它，因此没有采用 RFT 候选。每侧完整评测 1791 题。', '',
             '| 任务 | 原模型 正确/n（正确率） | 后训练 正确/n（正确率） | 绝对提升 | 配对提升95%区间 |',
             '|---|---:|---:|---:|---:|']
    for task in EXPECTED:
        b,c = metrics['base-final'][task],metrics['best-final'][task]
        d=deltas[task];lo,hi=d['paired_bootstrap_95_ci_pp']
        lines.append(f"| {titles[task]} | {b['correct']}/{b['n']}（{b['accuracy']*100:.2f}%） | {c['correct']}/{c['n']}（{c['accuracy']*100:.2f}%） | {d['delta_percentage_points']:+.2f} 个百分点 | [{lo:+.2f}, {hi:+.2f}] 个百分点 |")
    lines += ['', '所有逐题判定已重新汇总；两侧题号、题目、答案、测试和入口函数一致，无重复，与开发集题号及题目不重叠。输入文件 SHA256 已与远端核对。', '',
              '两侧采用同一 v2-hashlib-entrypoint 判分、math-verify 和隔离代码执行限制；seed=20261019、greedy、上下文2048、提示上限1100、最多生成1800 token且受剩余上下文限制。基座使用 continuation 提示，后训练使用 ChatML，渲染提示文本不同。', '',
              '这是冻结独立部分的完整结果，未用于继续选模或调参。不是整个原始基准的全部题目：开发集已提前隔离。数学答案匹配不代表每一步推理正确；代码通过只覆盖给定测试。', '',
              '区间使用固定随机种子的10000次配对自助抽样；是题目抽样不确定性，不能代表不同训练运行的波动。原模型为0时不报告相对倍数。', '',
              '本比较仅汇总既有隔离执行器/math-verify判定，没有重新生成答案或重新判分。完整哈希、准确率 Wilson 95%区间、逐题得失数量见同名 JSON。', '',
              '备份与停机状态单独核实；本报告生成本身不代表模型已完成本地备份或实例已停机。', '',
              '## 输入文件 SHA256', '']
    lines += [f"- `{name}`：`{v['sha256']}`" for name,v in inputs.items()]
    (a.artifacts/'final-eval-comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: result[k] for k in ['status','pipeline_finished_utc','metrics','deltas']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
