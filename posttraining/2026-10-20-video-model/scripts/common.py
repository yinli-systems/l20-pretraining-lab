"""Shared, reproducible model and chat formatting for the L20 reasoning run."""
import json, os, time, math
from pathlib import Path

ROOT = Path(os.environ.get('RUN_ROOT', '/workspace/posttrain-l20'))
CACHE = Path('/dev/shm/l20-posttrain')
MODEL_ID = 'AliceYin/L20-1B-20B-Base'
SEED = 20261019
TEMPLATE = "{% if bos_token %}{{ bos_token }}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

def json_safe(obj):
    if isinstance(obj,float) and not math.isfinite(obj): return None
    if isinstance(obj,dict): return {k:json_safe(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)): return [json_safe(v) for v in obj]
    return obj

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(json_safe(obj), indent=2, ensure_ascii=False, default=str,allow_nan=False))
    tmp.replace(path)

def log(event, **kwargs):
    row = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': event, **kwargs}
    print(json.dumps(json_safe(row), default=str,allow_nan=False), flush=True)

def tokenizer():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer', trust_remote_code=False)
    tok.model_input_names=['input_ids','attention_mask']
    tok.init_kwargs['model_input_names']=tok.model_input_names
    return tok

def encode_messages(tok, messages):
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    labels = [-100]*len(ids)
    head=tok.encode('<|im_start|>assistant\n',add_special_tokens=False)
    for j in range(len(ids)):
        if ids[j:j+len(head)]==head:
            k=j+len(head)
            while k<len(ids):
                labels[k]=ids[k]
                if ids[k]==tok.eos_token_id: break
                k+=1
    return ids, labels

def load_model(path=None, training=True, liger=True):
    import torch
    from transformers import AutoModelForCausalLM
    if training and liger:
        from liger_kernel.transformers import apply_liger_kernel_to_llama
        apply_liger_kernel_to_llama(rope=True, rms_norm=True, swiglu=True, cross_entropy=False, fused_linear_cross_entropy=True)
    source = path or str(CACHE/'base')
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32 if training else torch.bfloat16, attn_implementation='sdpa', trust_remote_code=False)
    tok = tokenizer()
    if model.config.vocab_size < len(tok):
        old_n = model.config.vocab_size
        model.resize_token_embeddings(len(tok), pad_to_multiple_of=64, mean_resizing=False)
        with torch.no_grad():
            model.get_input_embeddings().weight[old_n:].copy_(model.get_input_embeddings().weight[:old_n].mean(0))
            model.get_output_embeddings().weight[old_n:].copy_(model.get_output_embeddings().weight[:old_n].mean(0))
    model.config.eos_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = not training
    original_eos=tok.convert_tokens_to_ids('<|endoftext|>')
    model.generation_config.eos_token_id = list(dict.fromkeys([tok.eos_token_id,original_eos]))
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.suppress_tokens = list(range(len(tok),model.config.vocab_size)) or None
    return model.cuda()

def setup_torch():
    import random, numpy as np, torch
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch._inductor.config.compile_threads = 8

