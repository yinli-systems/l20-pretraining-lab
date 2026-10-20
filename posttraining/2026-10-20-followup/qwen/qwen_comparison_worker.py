"""Native-interface Qwen adapter around the frozen L20 generator and graders."""
import argparse
import concurrent.futures
import gc
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT=Path('/workspace/video-model')
JOB=ROOT/'comparisons/qwen-final-20261020'
sys.path.insert(0,str(ROOT))
import torch
import evaluate as frozen
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig


class NativeStops:
    def __init__(self,model,tok,stops):
        self.model,self.tok,self.stops=model,tok,stops
        self.config=model.config
    def eval(self):
        self.model.eval();return self
    def generate(self,**kwargs):
        assert kwargs['eos_token_id']==[self.tok.eos_token_id,0]
        kwargs['eos_token_id']=self.stops
        return self.model.generate(**kwargs)


def grade_code(pair):
    row,text=pair
    code=frozen.extract_code(text,row['prompt'],False,row['entry_point'])
    return {**row,'completion':text,'correct':frozen.code_correct(code,row['test'],row['entry_point'])}


def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--model',required=True);p.add_argument('--revision',required=True);a=p.parse_args()
    plan=json.loads((JOB/'plan.json').read_text())
    for name,digest in plan['frozen_sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((ROOT/'benchmarks.jsonl').read_bytes()).hexdigest()==plan['benchmark_sha256']
    frozen.setup_torch()
    rows=[x for x in map(json.loads,(ROOT/'benchmarks.jsonl').read_text().splitlines()) if x['split']=='final']
    from collections import Counter
    assert Counter(x['task'] for x in rows)=={'gsm8k':1223,'math500':436,'humaneval':132}
    assert len({(x['task'],x['id']) for x in rows})==1791
    profile=ROOT/'inference-profile.json'
    assert not profile.exists() or json.loads(profile.read_text()).get('selected',{}).get('mode')=='dynamic'
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=False)
    assert tok.chat_template
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tok.model_input_names=['input_ids','attention_mask']
    stop_tokens=['<|im_end|>','<|endoftext|>']
    stops=[tok.convert_tokens_to_ids(s) for s in stop_tokens]
    assert all(isinstance(i,int) and 0<=i<len(tok) and tok.convert_ids_to_tokens(i)==s for i,s in zip(stops,stop_tokens))
    assert tok.eos_token_id in stops
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa',trust_remote_code=False).cuda()
    model.generation_config=GenerationConfig.from_model_config(model.config)
    model.generation_config.eos_token_id=stops;model.generation_config.pad_token_id=tok.pad_token_id
    # Discard vendor sampling/repetition defaults: the original L20 run used
    # greedy decoding with one beam and the standard unit repetition penalty.
    model.generation_config.do_sample=False;model.generation_config.num_beams=1
    model.generation_config.repetition_penalty=1.0
    model.generation_config.no_repeat_ngram_size=0
    started=time.monotonic()
    outs=frozen.generate_rows(NativeStops(model,tok,stops),tok,rows,base=False,batch=32,max_new=1800,sample=False)
    del model;gc.collect();torch.cuda.empty_cache()
    pairs=list(zip(rows,outs));scored=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        scored.extend(ex.map(grade_code,[p for p in pairs if p[0]['task']=='humaneval']))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as ex:
        scored.extend(ex.map(frozen.grade_math_pair,[p for p in pairs if p[0]['task']!='humaneval']))
    metrics={}
    for task in ['gsm8k','math500','humaneval']:
        items=[r for r in scored if r['task']==task];k=sum(r['correct'] for r in items)
        metrics[task]={'correct':k,'n':len(items),'accuracy':k/len(items)}
    result={**plan['reference_protocol'],'model':next(m['repo'] for m in plan['models'] if m['key']==a.key),
            'model_revision':a.revision,'base_prompting':False,'metrics':metrics,
            'macro_accuracy':sum(m['accuracy'] for m in metrics.values())/3,
            'seconds':time.monotonic()-started,'eos_token_ids':stops,'eos_token_strings':stop_tokens,
            'cache_strategy':'dynamic','tokenizer':'native pinned Qwen tokenizer',
            'chat_template_sha256':hashlib.sha256(tok.chat_template.encode()).hexdigest(),
            'prompt_adaptation':plan['adaptation'],'generation_overrides':{'do_sample':False,'num_beams':1,'repetition_penalty':1.0,'no_repeat_ngram_size':0},
            'benchmark_sha256':plan['benchmark_sha256'],'frozen_sources':plan['frozen_sources']}
    sample=JOB/(a.key+'.samples.jsonl');tmp=sample.with_suffix('.jsonl.tmp')
    tmp.write_text(''.join(json.dumps(r)+'\n' for r in scored));tmp.replace(sample)
    frozen.write_json(JOB/(a.key+'.json'),result)
    frozen.log('evaluation_complete',**result)


if __name__=='__main__':main()
