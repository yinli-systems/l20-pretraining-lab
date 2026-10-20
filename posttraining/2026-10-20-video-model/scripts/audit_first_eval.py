"""CPU-only diagnostic; preserves existing scores and never changes a training job."""
import ast
import collections
import concurrent.futures
import datetime
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import CACHE, ROOT, SEED, encode_messages, tokenizer, write_json
from evaluate import extract_code, math_correct


def execute(code, row):
    payload = {"code": code, "test": row["test"], "entry_point": row["entry_point"], "jail": str(CACHE / "empty-jail")}
    try:
        result = subprocess.run([sys.executable, "-I", str(ROOT / "sandbox.py")], input=json.dumps(payload), text=True, capture_output=True, timeout=6)
        data = json.loads(result.stdout.strip().splitlines()[-1]) if result.stdout.strip() else {"passed": False, "error": "worker_exit"}
        return {**data, "returncode": result.returncode}
    except Exception as error:
        return {"passed": False, "error": type(error).__name__}


def entrypoint_code(text, row, base=False):
    """First complete fence defining the requested function, chosen without test feedback."""
    if base:
        return row["prompt"] + text
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    selected = None
    for block in blocks:
        try:
            tree = ast.parse(block)
        except (SyntaxError, ValueError):
            continue
        if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == row["entry_point"] for node in tree.body):
            selected = block
            break
    if selected is None:
        selected = extract_code(text)
    # The official completion protocol retains the prompt's imports and declarations.
    return row["prompt"] + "\n" + selected


def stage(name):
    print(json.dumps({"stage": name, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}), flush=True)


def main():
    logging.getLogger().setLevel(logging.ERROR)
    report = {"cpu_only": True, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "original_scores_preserved": True}
    tok = tokenizer()
    source = ROOT / "evaluations"
    datasets = {}
    for label, name in [("base", "base-dev"), ("sft2000", "sft-2000")]:
        path = source / (name + ".samples.jsonl")
        datasets[label] = [json.loads(line) for line in path.read_text().splitlines()]
    stage("packed_masks")
    mask_audit = {}
    rng = np.random.default_rng(SEED + 911)
    for split in ["train", "valid"]:
        ids = np.memmap(CACHE / "packed" / (split + ".ids.u16"), mode="r", dtype=np.uint16).reshape(-1, 2048)
        masks = np.memmap(CACHE / "packed" / (split + ".mask.u8"), mode="r", dtype=np.uint8).reshape(-1, 2048)
        indices = rng.choice(len(ids), min(256, len(ids)), replace=False)
        bad = []
        for index in indices:
            x = ids[index].tolist()
            _, labels = encode_messages(tok, [{"role": "user", "content": "mask audit"}, {"role": "assistant", "content": "answer"}])
            expected = np.zeros(2048, dtype=np.uint8)
            head = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
            for j in range(2048):
                if x[j:j + len(head)] == head:
                    k = j + len(head)
                    while k < 2048:
                        expected[k] = 1
                        if x[k] == tok.eos_token_id:
                            break
                        k += 1
            if not np.array_equal(expected, masks[index]):
                bad.append(int(index))
        mask_audit[split] = {"sampled_blocks": len(indices), "mismatched_blocks": bad}
    prepared = {}
    for name in ["math", "code", "numina", "general", "r1-math", "r1-code"]:
        p = CACHE / "prepared" / name
        offsets = np.load(p / "offsets.npy")
        ids = np.memmap(p / "ids.u16", mode="r", dtype=np.uint16)
        masks = np.memmap(p / "mask.u8", mode="r", dtype=np.uint8)
        samples = []
        for index in rng.choice(len(offsets) - 1, min(8, len(offsets) - 1), replace=False):
            a, b = int(offsets[index]), int(offsets[index + 1])
            x, m = ids[a:b], masks[a:b]
            samples.append({"tokens": b-a, "supervised_tokens": int(m.sum()), "prompt_prefix": tok.decode(x[m == 0].tolist())[:220], "assistant_prefix": tok.decode(x[m == 1].tolist())[:300], "assistant_suffix": tok.decode(x[m == 1].tolist())[-160:], "ends_with_assistant_eos": int(x[m == 1][-1]) == tok.eos_token_id})
        prepared[name] = samples
    report["mask_audit"] = mask_audit
    report["prepared_samples"] = prepared
    stage("math_grader")
    math_audit = {}
    for label, rows in datasets.items():
        result = collections.Counter()
        disagreements = []
        for row in rows:
            if row["task"] == "humaneval":
                continue
            oracle = math_correct("Final answer: \\boxed{" + row["answer"] + "}", row["answer"])
            actual = math_correct(row["completion"], row["answer"])
            result["n"] += 1
            result["oracle_passed"] += int(oracle)
            result["recomputed_passed"] += int(actual)
            result["boxed_outputs"] += int("\\boxed" in row["completion"])
            if actual != row["correct"]:
                disagreements.append(row["id"])
        math_audit[label] = {**result, "stored_score_disagreements": disagreements}
    report["math_grader"] = math_audit
    stage("code_grader")
    revision = json.loads((ROOT / "benchmark-provenance.json").read_text())["openai/openai_humaneval"]
    from datasets import load_dataset
    dev_ids = {r["id"] for r in datasets["base"] if r["task"] == "humaneval"}
    references = {r["task_id"]: r for r in load_dataset("openai/openai_humaneval", revision=revision, split="test") if r["task_id"] in dev_ids}
    oracle = []
    for key in sorted(dev_ids):
        ref = references[key]
        oracle.append({"id": key, **execute(ref["prompt"] + ref["canonical_solution"], ref)})
    code_audit = {}
    for label, rows in datasets.items():
        items = []
        for row in rows:
            if row["task"] != "humaneval":
                continue
            old = extract_code(row["completion"], row["prompt"], label == "base")
            revised = entrypoint_code(row["completion"], row, label == "base")
            old_result = execute(old, row)
            new_result = execute(revised, row)
            try:
                ast.parse(old)
                syntax = True
            except (SyntaxError, ValueError):
                syntax = False
            items.append({"id": row["id"], "stored_correct": row["correct"], "original": old_result, "entrypoint_parser": new_result, "original_syntax_valid": syntax, "extraction_changed": old != revised})
        code_audit[label] = {"n": len(items), "original_passed": sum(r["original"]["passed"] for r in items), "entrypoint_parser_passed": sum(r["entrypoint_parser"]["passed"] for r in items), "original_errors": dict(collections.Counter(r["original"].get("error", "passed") for r in items)), "items": items}
    report["code_reference_oracle"] = {"n": len(oracle), "passed": sum(r["passed"] for r in oracle), "items": oracle}
    report["code_grader"] = code_audit
    stage("export_checkpoint")
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    ready = json.loads((ROOT / "resume-ready.json").read_text())
    checkpoint_audit = {"resume_step": ready["step"], "selected_model": json.loads((ROOT / "best-model.json").read_text())}
    base_index = json.loads((CACHE / "base/model.safetensors.index.json").read_text())["weight_map"]
    best_index = json.loads((ROOT / "models/best/model.safetensors.index.json").read_text())["weight_map"]
    checkpoint_audit["same_parameter_names"] = set(base_index) == set(best_index)
    samples = []
    checkpoint = torch.load(ROOT / "resume.pt", mmap=True, map_location="cpu", weights_only=False)
    for name, shard in best_index.items():
        with safe_open(ROOT / "models/best" / shard, framework="pt", device="cpu") as handle:
            s = handle.get_slice(name)
            shape = s.get_shape()
            sl = tuple(slice(0, min(d, 8)) for d in shape)
            exported = s[sl]
        master = checkpoint["model"][name][sl]
        with safe_open(CACHE / "base" / base_index[name], framework="pt", device="cpu") as handle:
            original = handle.get_slice(name)[sl]
        samples.append({"name": name, "export_dtype": str(exported.dtype), "master_dtype": str(master.dtype), "finite": bool(torch.isfinite(exported).all()), "equals_resume_bf16": torch.equal(exported, master.to(torch.bfloat16)), "mean_abs_delta_from_base": float((exported.float() - original.float()).abs().mean())})
    del checkpoint
    checkpoint_audit["unchanged_resume_marker"] = ready == json.loads((ROOT / "resume-ready.json").read_text())
    checkpoint_audit["sampled_tensors"] = len(samples)
    checkpoint_audit["finite_tensors"] = sum(s["finite"] for s in samples)
    checkpoint_audit["sampled_export_resume_equal"] = sum(s["equals_resume_bf16"] for s in samples)
    checkpoint_audit["tensors_changed_from_base"] = sum(s["mean_abs_delta_from_base"] > 0 for s in samples)
    checkpoint_audit["tensor_samples"] = samples
    report["checkpoint_export"] = checkpoint_audit
    report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(ROOT / "audit-first-eval.json", report)
    stage("complete")
    print(json.dumps({"math_grader": math_audit, "code_reference_oracle_passed": report["code_reference_oracle"]["passed"], "code_grader": {k: {f: v for f, v in d.items() if f != "items"} for k, d in code_audit.items()}, "checkpoint": {k: v for k, v in checkpoint_audit.items() if k != "tensor_samples"}, "masks": mask_audit}), flush=True)


if __name__ == "__main__":
    main()
