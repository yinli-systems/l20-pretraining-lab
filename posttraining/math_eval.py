# Zero-shot chat math eval (MATH-500, GSM8K test): greedy, \boxed{} answer, graded with math-verify.
import sys, json
from datasets import load_dataset
from math_verify import parse, verify

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."

def gsm_gold(a): return a.split("####")[-1].strip().replace(",", "")

def run_math(llm, limit=None):
    from vllm import SamplingParams
    sets = {
        "math500": [(r["problem"], r["answer"]) for r in load_dataset("HuggingFaceH4/MATH-500", split="test")],
        "gsm8k_0shot": [(r["question"], gsm_gold(r["answer"])) for r in load_dataset("openai/gsm8k", "main", split="test")],
    }
    sp = SamplingParams(temperature=0, max_tokens=1536)
    res = {}
    for name, data in sets.items():
        data = data[:limit]
        outs = llm.chat([[{"role": "user", "content": q + SUFFIX}] for q, _ in data], sp, use_tqdm=False)
        correct = sum(bool(p := parse(o.outputs[0].text)) and verify(parse("$" + g + "$"), p)
                      for (_, g), o in zip(data, outs))
        res[name] = correct / len(data)
        print(f"{name} {res[name]:.4f}", flush=True)
    return res

def main():
    from vllm import LLM
    model, out = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    llm = LLM(model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.8, seed=0)
    json.dump(run_math(llm, limit), open(out, "w"))

if __name__ == "__main__":
    main()
