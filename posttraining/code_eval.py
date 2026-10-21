# HumanEval without an assistant prefill: chat prompt, greedy, take the first ```python block
# (or the whole reply), run the official tests in a subprocess. pass@1.
import sys, json, re, subprocess, tempfile, os
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset

PROMPT = "Complete the following Python function. Return the full function in a ```python code block.\n```python\n{}\n```"

def extract(text, prompt):
    m = re.search(r"```(?:python)?\n(.*?)(?:```|$)", text, re.S)
    code = m.group(1) if m else text
    # keep the prompt's imports/helpers if the model only returned the function
    return prompt + "\n" + code if "import" in prompt and "import" not in code else code

def run(program, timeout=10):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
    try:
        return subprocess.run([sys.executable, f.name], capture_output=True, timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(f.name)

def main():
    from vllm import LLM, SamplingParams
    model, out = sys.argv[1], sys.argv[2]
    ds = list(load_dataset("openai/openai_humaneval", split="test"))
    llm = LLM(model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.8, seed=0)
    outs = llm.chat([[{"role": "user", "content": PROMPT.format(d["prompt"])}] for d in ds],
                    SamplingParams(temperature=0, max_tokens=768), use_tqdm=False)
    progs = [extract(o.outputs[0].text, d["prompt"]) + "\n\n" + d["test"] + f"\ncheck({d['entry_point']})\n"
             for o, d in zip(outs, ds)]
    with ThreadPoolExecutor(32) as ex:
        ok = list(ex.map(run, progs))
    res = {"humaneval_chat_pass@1": sum(ok) / len(ok)}
    print("RESULT", json.dumps(res), flush=True)
    json.dump(res, open(out, "w"))

if __name__ == "__main__":
    main()
