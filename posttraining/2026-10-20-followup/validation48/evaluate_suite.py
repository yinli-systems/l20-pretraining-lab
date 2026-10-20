"""New custom-suite adapter; original generation and grading stay unmodified."""
import argparse
import collections
import concurrent.futures
import gc
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

OLD = Path('/workspace/video-model')
sys.path.insert(0, str(OLD))
import torch
import evaluate as frozen
from common import load_model, tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class NativeStops:
    def __init__(self, model, stops):
        self.model, self.config, self.stops = model, model.config, stops
    def eval(self):
        self.model.eval()
        return self
    def generate(self, **kwargs):
        kwargs['eos_token_id'] = self.stops
        return self.model.generate(**kwargs)


def grade_code(pair):
    row, text = pair
    code = frozen.extract_code(text, row['prompt'], False, row['entry_point'])
    return dict(row, completion=text, correct=frozen.code_correct(code, row['test'], row['entry_point']))


def retention(model, tok, rows):
    # Fixed next-letter likelihood protocol; never teacher-forced full answers.
    # All models must represent A..J as one token for this exact protocol.
    letters = 'ABCDEFGHIJ'
    choices = [tok.encode(x, add_special_tokens=False) for x in letters]
    if any(len(x) != 1 for x in choices):
        raise RuntimeError('Retention protocol needs a predeclared multi-token adapter')
    token_ids = [x[0] for x in choices]
    scored = []
    tok.padding_side = 'left'
    tok.truncation_side = 'left'
    for begin in range(0, len(rows), 32):
        batch = rows[begin:begin+32]
        prompts = []
        for row in batch:
            text = row['question'] + '\n\n' + '\n'.join(
                f'{letters[i]}. {option}' for i, option in enumerate(row['options']))
            text += '\n\nAnswer with the letter of the correct option.'
            prompts.append(tok.apply_chat_template([dict(role='user', content=text)],
                           tokenize=False, add_generation_prompt=True))
        encoded = tok(prompts, return_tensors='pt', padding=True, truncation=True,
                      max_length=1100, add_special_tokens=False).to('cuda')
        encoded.pop('token_type_ids', None)
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False).logits[:, -1, token_ids]
        for row, vector in zip(batch, logits.float().cpu().tolist()):
            pred = max(range(len(row['options'])), key=lambda i: vector[i])
            scored.append(dict(id=row['id'], category=row['category'], prediction=pred,
                               correct=pred == row['answer_index']))
    categories = sorted({r['category'] for r in scored})
    by_category = {c: sum(r['correct'] for r in scored if r['category'] == c) /
                  sum(r['category'] == c for r in scored) for c in categories}
    return dict(n=len(scored), correct=sum(r['correct'] for r in scored),
                category_macro_accuracy=sum(by_category.values())/len(categories),
                by_category=by_category, rows=scored,
                protocol='native chat, conditional next-token A..J likelihood, cap1100, no generation sampling')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--qwen', action='store_true')
    p.add_argument('--suite', required=True)
    p.add_argument('--retention')
    p.add_argument('--output', required=True)
    p.add_argument('--frozen-sources', required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() or out.with_suffix('.samples.jsonl').exists():
        raise RuntimeError('Existing evaluation outputs; do not rerun to select scores')
    out.parent.mkdir(parents=True, exist_ok=True)
    expected = json.loads(Path(args.frozen_sources).read_text())
    for name, sha in expected.items():
        if hashlib.sha256((OLD/name).read_bytes()).hexdigest() != sha:
            raise RuntimeError('Frozen grading/generation source changed')
    suite_path = Path(args.suite)
    rows = [json.loads(x) for x in suite_path.read_text().splitlines()]
    if len({r['id'] for r in rows}) != len(rows):
        raise RuntimeError('Duplicate suite IDs')
    frozen.setup_torch()
    if args.qwen:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa', trust_remote_code=False).cuda()
    else:
        tok = tokenizer()
        model = load_model(args.model, training=False, liger=False)
    tok.model_input_names = ['input_ids', 'attention_mask']
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    stops = [tok.convert_tokens_to_ids(s) for s in ['<|im_end|>', '<|endoftext|>']]
    if any(tok.convert_ids_to_tokens(i) != s for i, s in zip(stops, ['<|im_end|>', '<|endoftext|>'])):
        raise RuntimeError('Semantic EOS mapping failed')
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    model.generation_config.repetition_penalty = 1.
    model.generation_config.no_repeat_ngram_size = 0
    model.generation_config.eos_token_id = stops
    model.generation_config.pad_token_id = tok.pad_token_id
    if not args.qwen:
        model.generation_config.suppress_tokens = list(range(len(tok), model.config.vocab_size)) or None
    start = time.monotonic()
    texts = frozen.generate_rows(NativeStops(model, stops), tok, rows, batch=32, max_new=1800)
    keep = None
    if args.retention:
        keep_rows = [json.loads(x) for x in Path(args.retention).read_text().splitlines()]
        keep = retention(model, tok, keep_rows)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    pairs = list(zip(rows, texts))
    scored = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        scored.extend(ex.map(grade_code, [pair for pair in pairs if pair[0]['task'] == 'humaneval']))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as ex:
        scored.extend(ex.map(frozen.grade_math_pair, [pair for pair in pairs if pair[0]['task'] != 'humaneval']))
    domains = sorted({r['domain'] for r in scored})
    metrics = {}
    for domain in domains:
        items = [r for r in scored if r['domain'] == domain]
        correct = sum(r['correct'] for r in items)
        metrics[domain] = dict(correct=correct, n=len(items), accuracy=correct/len(items))
    result = dict(model=args.model, metrics=metrics,
                  macro_accuracy=sum(m['accuracy'] for m in metrics.values())/len(metrics),
                  seconds=time.monotonic()-start, suite_sha256=hashlib.sha256(suite_path.read_bytes()).hexdigest(),
                  source_sha256=expected, evaluation_protocol='v2-hashlib-entrypoint',
                  seed=20261019, context=2048, max_new_tokens=1800, prompt_cap=1100, generation='greedy',
                  native_chat_template_sha256=hashlib.sha256(tok.chat_template.encode()).hexdigest(),
                  native_eos_ids=stops, actual_padded_prompt_limits=True, retention=keep,
                  note='Custom question domains, not official GSM8K/MATH500/HumanEval scores.')
    samples = out.with_suffix('.samples.jsonl')
    temp = samples.with_suffix('.jsonl.tmp')
    temp.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in scored))
    temp.replace(samples)
    frozen.write_json(out, result)


if __name__ == '__main__':
    main()
