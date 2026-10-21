# All benchmarks in ONE process with one vLLM engine, so the GPU is never released between tasks.
# usage: CUDA_VISIBLE_DEVICES=g python eval_all.py <model> <outdir> [chat|base] [--wait-for PATH]
import sys, os, json, time, subprocess

def wait_gpu(path, need_gb=36):
    # Wait until `path` exists, then until the visible GPU has room, then start immediately.
    while path and not os.path.exists(path):
        time.sleep(5)
    while True:
        free = int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             "-i", os.environ["CUDA_VISIBLE_DEVICES"]]).decode().strip())
        if free > need_gb * 1024:
            return
        time.sleep(0.3)

def main():
    model, out, mode = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "chat")
    wait = sys.argv[sys.argv.index("--wait-for") + 1] if "--wait-for" in sys.argv else None
    if wait:
        wait_gpu(wait)
    os.makedirs(out, exist_ok=True)
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    import lm_eval
    from lm_eval.models.vllm_causallms import VLLM
    from math_eval import run_math
    lm = VLLM(pretrained=model, dtype="bfloat16", gpu_memory_utilization=0.8, max_model_len=2048,
              enable_prefix_caching=True, batch_size="auto", seed=0)
    chat = mode == "chat"
    part = os.environ.get("PART", "all")  # all | math | gen | mmlu | cs  (split across GPUs)
    summary = {}
    if chat and part in ("all", "math"):
        summary.update(run_math(lm.model))
        print(json.dumps(summary), flush=True)
    runs = [
        dict(tasks=["gsm8k"], num_fewshot=5, gen_kwargs="max_gen_toks=512"),
        dict(tasks=["minerva_math"], num_fewshot=4, gen_kwargs="max_gen_toks=512"),
    ]
    if chat:
        runs += [dict(tasks=["ifeval"], gen_kwargs="max_gen_toks=1024"),
                 dict(tasks=["humaneval_instruct"], gen_kwargs="max_gen_toks=768", confirm_run_unsafe_code=True)]
    else:
        runs += [dict(tasks=["humaneval"], confirm_run_unsafe_code=True)]
    runs += [dict(tasks=["mmlu"], num_fewshot=5, loglik=True),
             dict(tasks=["arc_challenge", "arc_easy", "hellaswag", "piqa", "winogrande", "openbookqa", "truthfulqa_mc2"], loglik=True)]
    groups = {"math": ["gsm8k"], "gen": ["ifeval", "humaneval_instruct", "humaneval"], "mmlu": ["mmlu"],
              "cs": ["arc_challenge", "minerva_math"]}
    if part != "all":
        runs = [r for r in runs if r["tasks"][0] in groups[part] and r["tasks"][0] != "minerva_math"]
    for r in runs:
        loglik = r.pop("loglik", False)
        use_chat = chat and not loglik
        res = lm_eval.simple_evaluate(model=lm, apply_chat_template=use_chat,
                                      fewshot_as_multiturn=use_chat and "num_fewshot" in r, random_seed=0, **r)
        for t, v in res["results"].items():
            summary[t] = {k: v[k] for k in v if isinstance(v[k], float) and "stderr" not in k}
        print(lm_eval.utils.make_table(res), flush=True)
        json.dump(summary, open(f"{out}/summary_{part}.json", "w"), indent=1)
    print("EVAL_DONE", out, flush=True)

if __name__ == "__main__":
    main()
