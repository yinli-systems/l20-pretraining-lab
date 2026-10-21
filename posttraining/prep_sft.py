# Build packed SFT data: ChatML, loss on assistant turns only, 2048-token rows.
import glob, os, random, json, numpy as np
from multiprocessing import Pool
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

ROOT = "/workspace/l20-posttrain"
HUB = os.environ["HF_HOME"].rstrip("/") + "/hub"
SEQ = 2048
random.seed(0)

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{% if m['role'] == 'assistant' %}{% generation %}{{ m['content'] }}<|im_end|>{% endgeneration %}"
    "{% else %}{{ m['content'] }}<|im_end|>{% endif %}\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

def make_init():
    out = f"{ROOT}/init"
    tok = AutoTokenizer.from_pretrained(f"{ROOT}/base")
    tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    tok.chat_template = CHAT_TEMPLATE
    tok.eos_token = "<|im_end|>"
    if os.path.exists(f"{out}/config.json"):
        return tok
    m = AutoModelForCausalLM.from_pretrained(f"{ROOT}/base", torch_dtype=torch.float32)
    m.resize_token_embeddings(len(tok), pad_to_multiple_of=64, mean_resizing=True)
    m.config.eos_token_id = tok.convert_tokens_to_ids("<|im_end|>")
    m.generation_config.eos_token_id = [tok.convert_tokens_to_ids("<|im_end|>"), 0]
    m.generation_config.pad_token_id = 1
    m.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    return tok

def files(pattern):
    return sorted(glob.glob(f"{HUB}/{pattern}", recursive=True))

def load_sources():
    rows = []  # (source, messages)
    for f in files("datasets--HuggingFaceTB--smoltalk/snapshots/*/data/all/train-*.parquet"):
        t = pq.read_table(f).to_pylist()
        rows += [("smoltalk/" + r["source"], r["messages"]) for r in t]
    omi = []
    for f in files("datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"):
        omi += pq.read_table(f, columns=["problem", "generated_solution"]).to_pylist()
    random.shuffle(omi)
    rows += [("omi2", [{"role": "user", "content": r["problem"]},
                       {"role": "assistant", "content": r["generated_solution"]}]) for r in omi[:600_000]]
    for f in files("datasets--OpenCoder-LLM--opc-sft-stage2/snapshots/*/**/*.parquet"):
        name = f.split("/")[-2]
        t = pq.read_table(f, columns=["instruction", "output"]).to_pylist()
        rows += [("opc/" + name, [{"role": "user", "content": r["instruction"]},
                                   {"role": "assistant", "content": r["output"]}]) for r in t]
    for f in files("datasets--allenai--tulu-3-sft-personas-instruction-following/snapshots/*/data/*.parquet"):
        t = pq.read_table(f, columns=["messages"]).to_pylist()
        rows += [("tulu_if", r["messages"]) for r in t]
    return rows

TOK = None
def _init(path):
    global TOK, IMS, IME, NL
    TOK = AutoTokenizer.from_pretrained(path)
    IMS, IME = TOK.convert_tokens_to_ids("<|im_start|>"), TOK.convert_tokens_to_ids("<|im_end|>")
    NL = TOK.encode("\n", add_special_tokens=False)

def encode(item):
    src, msgs = item
    ids, lab = [], []
    has_asst = False
    for m in msgs:
        role, content = m["role"], m["content"] or ""
        head = [IMS] + TOK.encode(role + "\n", add_special_tokens=False)
        body = TOK.encode(content, add_special_tokens=False) + [IME]
        ids += head; lab += [-100] * len(head)
        if role == "assistant":
            ids += body; lab += body; has_asst = True
        else:
            ids += body; lab += [-100] * len(body)
        ids += NL; lab += [-100] * len(NL)
        if len(ids) > SEQ:
            return None
    if not has_asst:
        return None
    return src, np.array(ids, np.int32), np.array(lab, np.int32)

def pack(samples):
    # first-fit-decreasing into SEQ-length bins
    order = sorted(range(len(samples)), key=lambda i: -len(samples[i][1]))
    # best-fit: bucket open bins by remaining space
    bins, space = [], []
    free = [[] for _ in range(SEQ + 1)]
    cnt = np.zeros(SEQ + 1, np.int32)
    for i in order:
        L = len(samples[i][1])
        nz = np.flatnonzero(cnt[L:])
        if len(nz):
            r = L + nz[0]; placed = free[r].pop(); cnt[r] -= 1
        else:
            bins.append([]); space.append(SEQ); placed = len(bins) - 1
        bins[placed].append(i); space[placed] -= L
        free[space[placed]].append(placed); cnt[space[placed]] += 1
    return bins

def main():
    tok = make_init()
    rows = load_sources()
    print("raw rows", len(rows), flush=True)
    with Pool(64, initializer=_init, initargs=(f"{ROOT}/init",)) as p:
        enc = [e for e in p.imap(encode, rows, chunksize=512) if e is not None]
    print("kept", len(enc), flush=True)
    stats = {}
    for s, i, l in enc:
        k = s.split("/")[0]
        st = stats.setdefault(k, [0, 0, 0]); st[0] += 1; st[1] += len(i); st[2] += int((l != -100).sum())
    print(json.dumps(stats, indent=1))
    random.shuffle(enc)
    bins = pack(enc)
    random.shuffle(bins)
    N = len(bins)
    ids = np.lib.format.open_memmap(f"{ROOT}/data/sft_ids.npy", "w+", np.int32, (N, SEQ))
    lab = np.lib.format.open_memmap(f"{ROOT}/data/sft_lab.npy", "w+", np.int32, (N, SEQ))
    pos = np.lib.format.open_memmap(f"{ROOT}/data/sft_pos.npy", "w+", np.int16, (N, SEQ))
    for b, members in enumerate(bins):
        o = 0
        for i in members:
            _, x, y = enc[i]; L = len(x)
            ids[b, o:o + L] = x; lab[b, o:o + L] = y; pos[b, o:o + L] = np.arange(L); o += L
        # pad tail as its own pseudo-doc, ignored by loss
        ids[b, o:] = 1; lab[b, o:] = -100; pos[b, o:] = np.arange(SEQ - o)
    ids.flush(); lab.flush(); pos.flush()
    tot = sum(len(e[1]) for e in enc)
    print(f"rows {N} fill {tot / (N * SEQ):.3f} tokens {tot/1e9:.3f}B", flush=True)

if __name__ == "__main__":
    main()
