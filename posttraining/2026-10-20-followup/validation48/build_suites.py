"""Collect/decontaminate new suite proposals; no GPU or ready marker.

ASDiv annotations are recomputed with a restricted arithmetic AST. Competition
labels retain upstream references and explicitly await independent review.
Python questions use the already execution-checked candidate references and
split all identical canonical AST families together. Proposals freeze only
after a separate recorded review, before any next-phase model score is seen.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import ast
import collections
import concurrent.futures
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from screening import JOB,WORK,PriorScreen,build_fts,norm,digest,family,code_family


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def arithmetic(text):
    text=text.strip().replace(',','').replace('×','*').replace('÷','/').replace('^','**')
    if not re.fullmatch(r'[0-9eE.+*/() \-]+',text):raise ValueError('Not restricted arithmetic')
    tree=ast.parse(text,mode='eval')
    def visit(node):
        if isinstance(node,ast.Expression):return visit(node.body)
        if isinstance(node,ast.Constant) and type(node.value) in [int,float]:return Fraction(str(node.value))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            return -visit(node.operand) if isinstance(node.op,ast.USub) else visit(node.operand)
        if isinstance(node,ast.BinOp):
            a,b=visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            if isinstance(node.op,ast.Div):return a/b
            if isinstance(node.op,ast.Pow) and b.denominator==1 and abs(b)<20:return a**int(b)
        raise ValueError('Unsupported arithmetic operator')
    value=visit(tree)
    if abs(value)>10**15:raise ValueError('Unbounded numeric label')
    return value


def word_candidates(screen):
    source=JOB/'suite-acquisition/ASDiv.xml';root=ET.fromstring(source.read_bytes());rows=[];counts=collections.Counter()
    provenance=json.loads((JOB/'suite-acquisition/asdiv-source.json').read_text())
    for problem in root.findall('.//Problem'):
        counts['input']+=1
        p=(problem.findtext('Body','')+' '+problem.findtext('Question','')).strip()
        try:
            formula=problem.findtext('Formula','').strip();sides=formula.split('=')
            if len(sides)!=2:raise ValueError()
            left,right=map(arithmetic,sides)
            answer=problem.findtext('Answer','').split('(')[0].strip()
            gold=arithmetic(answer)
            if left!=right or right!=gold:raise ValueError()
        except Exception:counts['non_scalar_or_inconsistent_annotation']+=1;continue
        if screen.matches(p):counts['prior_overlap']+=1;continue
        rows.append(dict(id='asdiv-'+problem.attrib['ID'],task='gsm8k',domain='arithmetic_word_problems',
              prompt=p,answer=str(gold),test='',entry_point='',source='ASDiv',source_revision=provenance['revision'],
              family=family(p),grade=problem.attrib['Grade'],solution_type=problem.findtext('Solution-Type',''),
              label_validation=dict(kind='human annotation plus independent restricted-Fraction formula recomputation',formula=formula),
              reference=problem.findtext('Answer',''),license='CC-BY-NC-4.0 (evaluation-only)'))
    write(JOB/'suite-acquisition/word-eligibility.json',dict(counts=dict(counts),eligible=len(rows)))
    return rows


def competition_candidates(screen):
    from datasets import load_dataset
    from math_verify import parse,LatexExtractionConfig
    revision='40ba231d8f16e29ecd40e6407e2c8640145a8f62'
    ds=load_dataset('KbsdJames/Omni-MATH',split='test',revision=revision,streaming=True)
    rows=[];counts=collections.Counter()
    for row in ds:
        counts['input']+=1
        answer=row.get('answer','')
        try:
            expr=parse('\\boxed{'+answer+'}',extraction_config=[LatexExtractionConfig()])
            if not expr or not getattr(expr[0],'is_number',False) or getattr(expr[0],'free_symbols',None):
                raise ValueError()
        except Exception:counts['not_closed_scalar']+=1;continue
        p=row['problem']
        if screen.matches(p):counts['prior_overlap']+=1;continue
        rows.append(dict(id='omni-'+digest(p),task='math500',domain='competition_style_math',prompt=p,
             answer=answer,test='',entry_point='',family=family(p),difficulty=row['difficulty'],
             source='KbsdJames/Omni-MATH',source_revision=revision,source_category=row['source'],
             label_validation=dict(kind='upstream competition answer; scalar parser control; independent gold review pending'),
             reference=row['solution'],license='Apache-2.0',review_pending=True))
    write(JOB/'suite-acquisition/competition-eligibility.json',dict(counts=dict(counts),eligible=len(rows)))
    return rows


def code_stub(row):
    tree=ast.parse(row['reference']);entry=row['entry_point']
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==entry)
    # Expose the function signature and task only, without a reference body.
    fn.body=[ast.Expr(value=ast.Constant(value=row['prompt']))]
    fn.returns=None;fn.decorator_list=[]
    for arg in list(fn.args.posonlyargs)+list(fn.args.args)+list(fn.args.kwonlyargs):arg.annotation=None
    if fn.args.vararg:fn.args.vararg.annotation=None
    if fn.args.kwarg:fn.args.kwarg.annotation=None
    imports='from typing import *\nimport math, collections, itertools, functools, heapq, bisect, re, string, statistics, fractions, decimal, operator, copy, hashlib\n'
    stub=imports+ast.unparse(fn)+'\n'
    # Default expressions may refer to unavailable globals. Exclude such tasks
    # instead of supplying a reference helper as a hint.
    for default in fn.args.defaults+list(x for x in fn.args.kw_defaults if x is not None):
        if any(isinstance(x,(ast.Call,ast.Name,ast.Attribute)) for x in ast.walk(default)):raise ValueError('Unresolved function default')
    return stub


def code_candidates(screen):
    sys.path.insert(0,'/workspace/video-model')
    from evaluate import code_correct,extract_code
    source=WORK/'candidates/code.jsonl'
    if not source.exists():return []
    data=source.read_bytes();data=data[:data.rfind(b'\n')+1]
    rows=[];seen=set();counts=collections.Counter()
    old=WORK/'prior-code-families.sqlite'
    if not (JOB/'prior-code-families-complete.json').exists():
        write(JOB/'suite-acquisition/code-eligibility.json',dict(status='waiting_for_old_AST_index',ready=False))
        return []
    db=sqlite3.connect('file:'+str(old)+'?mode=ro',uri=True)
    for line in data.splitlines():
        row=json.loads(line);counts['input']+=1
        try:
            fam=code_family(row['reference'])
            if fam in seen or db.execute('SELECT 1 FROM families WHERE hash=?',(fam,)).fetchone():
                counts['old_or_duplicate_AST_family']+=1;continue
            # Assign all members of a canonical AST family to the same role,
            # before any score. Holdout pool is 30%, training is the other70%.
            bucket=int(fam[:12],16)%10000
            if bucket>=3000:continue
            if screen.matches(row['prompt']):counts['prior_prompt_overlap']+=1;continue
            stub=code_stub(row)
            positive=extract_code('```python\n'+row['reference']+'\n```',stub,False,row['entry_point'])
            test='def check(candidate):\n'+''.join('    '+x+'\n' for x in row['test'].splitlines())
            # The assertions already reference the original function name;
            # target is passed to the same check wrapper as the frozen grader.
            if not code_correct(positive,test,row['entry_point']):
                counts['stub_reference_control_failed']+=1;continue
            if code_correct(stub,test,row['entry_point']):
                counts['empty_stub_passed']+=1;continue
            seen.add(fam)
            rows.append(dict(id='oci-'+row['source_id'],task='humaneval',domain='python_functions',prompt=stub,
                 answer='',test=test,entry_point=row['entry_point'],family=fam,
                 assigned_split='dev' if bucket<1000 else 'confirmation',
                 source=row['source'],source_revision=row['revision'],license='CC-BY-4.0',
                 source_prompt=row['prompt'],reference=row['reference'],
                 label_validation=dict(kind='reference replay, >=3 distinct cases with boundary; None/0/empty controls rejected; exact frozen stub execution also verified')))
        except Exception:counts['format_rejected']+=1
    db.close()
    write(JOB/'suite-acquisition/code-eligibility.json',dict(counts=dict(counts),eligible=len(rows),
         eligible_by_role=dict(collections.Counter(r['assigned_split'] for r in rows))))
    return rows


def retention_candidates(screen):
    from datasets import load_dataset
    revision='b189ec765aa7ed75c8acfea42df31fdae71f97be'
    categories=['business','economics','health','history','law','philosophy','psychology','other']
    ds=load_dataset('TIGER-Lab/MMLU-Pro',split='test',revision=revision,streaming=True)
    groups=collections.defaultdict(list)
    for r in ds:
        if r['category'] not in categories:continue
        text=r['question']+' '+' '.join(r['options'])
        if screen.matches(text):continue
        groups[r['category']].append(dict(id='mmlu-pro-'+str(r['question_id']),question=r['question'],
             options=r['options'],answer_index=r['answer_index'],category=r['category'],family=family(text),
             source='TIGER-Lab/MMLU-Pro',source_revision=revision,license='MIT',role='retention-only general knowledge development probe'))
    rows=[]
    for category in categories:
        candidates=sorted(groups[category],key=lambda r:digest('retention20261020'+r['id']))
        if len(candidates)<125:raise RuntimeError('Insufficient retention questions: '+category)
        rows.extend(candidates[:125])
    return rows


def split_math(rows,domain):
    # Hash ordering and one representative per normalized family; no model
    # performance enters selection. Keep source/difficulty for cluster reports.
    rows=sorted(rows,key=lambda r:digest('suite20261020'+r['id']))
    unique={}
    for row in rows:unique.setdefault(row['family'],row)
    rows=list(unique.values())
    if len(rows)<1200:raise RuntimeError(f'{domain}: only {len(rows)} eligible families; need1200')
    return [dict(r,assigned_split='dev' if i<400 else 'confirmation') for i,r in enumerate(rows[:1200])]


def main():
    p=argparse.ArgumentParser();p.add_argument('--deadline',type=float,required=True);args=p.parse_args()
    (JOB/'suite-acquisition').mkdir(exist_ok=True)
    index_child=None
    index_log=None
    if not (JOB/'prior-code-families-complete.json').exists():
        index_log=(JOB/'prior-code-families.log').open('a')
        index_child=subprocess.Popen([sys.executable,str(JOB/'prior_code_families.py'),'--deadline',str(args.deadline)],stdout=index_log,stderr=subprocess.STDOUT)
    if not (JOB/'prior-fts-complete.json').exists():build_fts(args.deadline)
    screen=PriorScreen()
    word=word_candidates(screen);competition=competition_candidates(screen);keep=retention_candidates(screen)
    write(JOB/'suite-acquisition/math-proposals.json',dict(word=word,competition=competition))
    write(JOB/'suite-acquisition/retention-proposal.json',keep)
    while time.time()<args.deadline:
        if index_child is not None and index_child.poll() is not None and index_child.returncode:
            raise RuntimeError('Prior AST index failed; inspect log')
        code=code_candidates(screen)
        by_role=collections.Counter(r['assigned_split'] for r in code)
        status=dict(status='collecting_suite_proposals',word=len(word),competition=len(competition),code_by_role=dict(by_role),ready_for_evaluation=False)
        write(JOB/'suite-status.json',status)
        if len(word)>=1200 and len(competition)>=1200 and by_role['dev']>=400 and by_role['confirmation']>=800:
            rows=split_math(word,'word')+split_math(competition,'competition')
            for role,n in [('dev',400),('confirmation',800)]:
                pool=sorted((r for r in code if r['assigned_split']==role),key=lambda r:digest('code20261020'+r['id']))
                rows.extend(pool[:n])
            for role in ['dev','confirmation']:
                path=JOB/'suite-acquisition'/f'proposed-{role}.jsonl'
                path.write_text(''.join(json.dumps(dict(r,split=role),ensure_ascii=False)+'\n' for r in rows if r['assigned_split']==role))
            write(JOB/'suite-status.json',dict(status='proposals_complete_review_required',ready_for_evaluation=False,
                dev=1200,confirmation=2400,retention=1000,scope='custom domains, not official benchmark scores',
                pending=['independent gold/source200 review','final code-family and numeric-family split cross-check','immutable suite freeze']))
            return
        time.sleep(120)
    raise TimeoutError('Suite proposal capacity not satisfied within preparation bound')


if __name__=='__main__':main()
