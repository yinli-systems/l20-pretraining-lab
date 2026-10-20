import json, os, sys
from pathlib import Path
from common import ROOT, CACHE, MODEL_ID, TEMPLATE, write_json, log

def main():
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer
    ROOT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    prior=ROOT/'model-provenance.json'
    revision=json.loads(prior.read_text())['revision'] if prior.exists() else HfApi().model_info(MODEL_ID).sha
    write_json(prior, {'model': MODEL_ID, 'revision': revision, 'downloaded_as': 'unmodified FP32 safetensors'})
    snapshot_download(MODEL_ID, revision=revision, local_dir=CACHE/'base', allow_patterns=['*.json','*.safetensors','*.model','*.txt'])
    tok = AutoTokenizer.from_pretrained(CACHE/'base', trust_remote_code=False)
    orig = {'length':len(tok),'bos':tok.bos_token,'eos':tok.eos_token,'pad':tok.pad_token}
    tok.add_special_tokens({'additional_special_tokens':['<|im_start|>','<|im_end|>']})
    tok.eos_token = '<|im_end|>'
    if tok.pad_token is None: tok.pad_token = orig['eos'] or '<|im_end|>'
    tok.chat_template = TEMPLATE
    tok.model_input_names=['input_ids','attention_mask']
    tok.init_kwargs['model_input_names']=tok.model_input_names
    tok.save_pretrained(ROOT/'tokenizer')
    write_json(ROOT/'tokenizer-changes.json', {'original':orig,'length':len(tok),'bos_id':tok.bos_token_id,'eos_id':tok.eos_token_id,'pad_id':tok.pad_token_id,'template':TEMPLATE})
    log('bootstrap_complete', model_revision=revision, tokenizer_length=len(tok))

if __name__=='__main__': main()
