"""Isolated full-parameter pilot stage; never writes the historical run.

Launch only after the controller has persisted readiness and the 48h budget.
Data identities, schedules and cursor/RNG state are bound into every recovery.
"""
import argparse
import contextlib
import datetime as dt
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from objective import CONTEXT, Packed, ResumableBatches, file_sha, optimizer_step_budget

OLD = Path('/workspace/video-model')
JOB = OLD / 'validation-48h'
WORK = Path('/dev/shm/video-model-validation48')
sys.path.insert(0, str(OLD))
from common import load_model, tokenizer, write_json, log


def atomic_checkpoint(model, optimizer, state, path):
    # Full FP32 master weights and Adam states, never the old RFT optimizer.
    temp = path.with_suffix('.next.pt')
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), state=state,
                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                   numpy_rng=np.random.get_state(), python_rng=random.getstate())
    expected = sum(t.numel()*t.element_size() for t in payload['model'].values())
    expected += sum(t.numel()*t.element_size() for s in payload['optimizer']['state'].values()
                    for t in s.values() if torch.is_tensor(t)) + 1024**2
    if shutil.disk_usage(path.parent).free < expected + 512*1024**2:
        raise RuntimeError('Recovery capacity gate: preserve existing full checkpoint; no partial replacement attempted')
    torch.save(payload, temp)
    with temp.open('rb') as stream:
        os.fsync(stream.fileno())
    temp.replace(path)
    write_json(path.with_suffix('.json'), dict(stage=state['stage'], step=state['step'],
               bytes=path.stat().st_size, sha256=file_sha(path), checkpoint=str(path),
               data_identity=state['data_identity'], config_identity=state['config_identity']))


def export(model, path):
    temp = path.with_name(path.name + '-next')
    if temp.exists():
        raise RuntimeError('Incomplete prior export requires explicit recovery review')
    weights = {k: v.detach().to(device='cpu', dtype=torch.bfloat16).contiguous()
               for k, v in model.state_dict().items()}
    model.save_pretrained(temp, state_dict=weights, safe_serialization=True, max_shard_size='2GB')
    tokenizer().save_pretrained(temp)
    del weights
    temp.rename(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    identity = file_sha(args.config)
    stage = cfg['stage']
    if stage not in ['B1', 'B2', 'A-cpt', 'A-bridge', 'repeat-B1', 'repeat-B2', 'repeat-A-cpt', 'repeat-A-bridge']:
        raise ValueError('Unapproved pilot stage')
    readiness = json.loads((JOB / 'readiness.json').read_text())
    budget = json.loads((JOB / 'budget.json').read_text())
    if not readiness['ready'] or budget['max_gpu_seconds'] != 48 * 3600:
        raise RuntimeError('Readiness or authorized budget gate failed')
    if budget['readiness_sha256'] != file_sha(JOB / 'readiness.json'):
        raise RuntimeError('Readiness changed after budget freeze')
    if budget['deadline_unix'] - budget['started_unix'] != 48 * 3600:
        raise RuntimeError('Unexpected budget extension')
    deadline = budget['deadline_unix']
    if time.time() >= deadline - 900:
        raise RuntimeError('Less than 15 minutes remain; do not start a stage')
    WORK.mkdir(parents=True, exist_ok=True)
    lock = (JOB / 'gpu-stage.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = JOB / 'stages' / stage
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'complete.json').exists():
        raise RuntimeError('Stage already completed; refusing duplicate training')
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch._inductor.config.compile_threads = 8
    ds = Packed(cfg['data'], cfg['objective'])
    b, accum = cfg['micro_batch'], cfg['accumulation']
    steps = optimizer_step_budget(len(ds), b, accum, cfg['token_cap'], cfg.get('max_epochs'))
    cfg_snapshot = dict(cfg, actual_planned_steps=steps, data_identity=ds.identity,
                        config_identity=identity, actual_init_hashes=cfg['init_sha256'])
    write_json(output / 'config-frozen.json', cfg_snapshot)
    init = Path(cfg['init'])
    for name, sha in cfg['init_sha256'].items():
        if file_sha(init / name) != sha:
            raise RuntimeError('Initial weights/config identity mismatch: ' + name)
    model = load_model(str(init), training=True)
    # At a stage handoff the last completed optimizer remains protected until
    # this endpoint has a verified local backup. Reserve both this stage's full
    # recovery and its atomic replacement, plus its BF16 export and headroom.
    model_bytes=sum(p.numel()*p.element_size() for p in model.state_dict().values())
    recovery_bytes=3*model_bytes+1024**2
    required=(recovery_bytes if args.resume else 2*recovery_bytes)+model_bytes//2+1536*1024**2
    free=shutil.disk_usage(WORK).free
    write_json(output/'storage-preflight.json',dict(free_bytes=free,required_bytes=required,
        recovery_estimated_bytes=recovery_bytes,model_fp32_bytes=model_bytes,
        preserves_prior_completed_recovery=True,passed=free>=required))
    if free<required:
        raise RuntimeError('Stage capacity gate failed before optimizer/training; original and prior recoveries retained')
    model.train()
    if cfg.get('compiled', True):
        model.model.forward = torch.compile(model.model.forward, dynamic=False, mode='default')
    groups = [dict(params=[p for p in model.parameters() if p.ndim >= 2], weight_decay=cfg['weight_decay']),
              dict(params=[p for p in model.parameters() if p.ndim < 2], weight_decay=0.)]
    optimizer = torch.optim.AdamW(groups, lr=cfg['lr'], betas=(.9, .95), eps=1e-8, fused=True)
    state = dict(stage=stage, step=0, consumed=0, tokens=0, supervised_tokens=0,
                 seed=seed, data_identity=ds.identity, config_identity=identity,
                 planned_steps=steps, initial_weights=cfg['init_sha256'], deadline_unix=deadline)
    checkpoint = WORK / 'recovery' / (stage + '.pt')
    checkpoint.parent.mkdir(exist_ok=True)
    if args.resume:
        saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
        old = saved['state']
        for key in ['stage', 'data_identity', 'config_identity', 'planned_steps', 'initial_weights', 'deadline_unix']:
            if old[key] != state[key]:
                raise RuntimeError('Recovery identity mismatch: ' + key)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        state = old
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        del saved
    sampler = ResumableBatches(len(ds), b, seed, state['consumed'])
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=2, pin_memory=True,
                    persistent_workers=True, prefetch_factor=3)
    iterator = iter(dl)
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    window = time.monotonic()
    packed_window = supervised_window = 0
    warmup = max(1, int(steps * cfg['warmup_fraction']))
    from torch.nn.attention import sdpa_kernel, SDPBackend
    backend = cfg.get('attention_backend', 'cudnn')
    attention = sdpa_kernel(SDPBackend.CUDNN_ATTENTION) if backend == 'cudnn' else contextlib.nullcontext()
    with (output / 'metrics.jsonl').open('a', buffering=1) as metrics, attention:
        while state['step'] < steps and time.time() < deadline - 600 and not stop:
            batches = []
            for _ in range(accum):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(dl)
                    batch = next(iterator)
                batches.append(batch)
                sampler.consumed += 1
            count = sum(int(batch[2].sum()) for batch in batches)
            if not count:
                raise RuntimeError('Empty gradient accumulation window')
            step = state['step']
            scale = (step + 1) / warmup if step < warmup else .1 + .45 * (1 + math.cos(
                math.pi * min(1., (step - warmup) / max(1, steps - warmup))))
            for group in optimizer.param_groups:
                group['lr'] = cfg['lr'] * scale
            batch_loss = 0.
            for ids, labels, n in batches:
                weight = int(n.sum()) / count
                ids = ids.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(input_ids=ids, labels=labels, use_cache=False).loss
                (loss * weight).backward()
                batch_loss += float(loss.detach()) * weight
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
            if not torch.isfinite(norm) or not math.isfinite(batch_loss):
                raise RuntimeError('Nonfinite loss/gradient; stop this candidate')
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            packed = b * accum * CONTEXT
            state.update(step=step+1, consumed=sampler.consumed, tokens=state['tokens']+packed,
                         supervised_tokens=state['supervised_tokens']+count)
            packed_window += packed
            supervised_window += count
            if state['step'] == 1 or state['step'] % 20 == 0:
                torch.cuda.synchronize()
                elapsed = time.monotonic() - window
                stats = dict(state, loss=batch_loss, gradient_norm=float(norm),
                             lr_current=cfg['lr']*scale,
                             packed_tokens_per_second=packed_window/elapsed,
                             supervised_tokens_per_second=supervised_window/elapsed,
                             utc=dt.datetime.now(dt.timezone.utc).isoformat())
                metrics.write(json.dumps(stats) + '\n')
                write_json(output / 'status.json', stats)
                log('pilot_training', **stats)
                window = time.monotonic()
                packed_window = supervised_window = 0
            if state['step'] % 500 == 0:
                atomic_checkpoint(model, optimizer, state, checkpoint)
                write_json(JOB / 'latest-recovery.json', json.loads(checkpoint.with_suffix('.json').read_text()))
        atomic_checkpoint(model, optimizer, state, checkpoint)
        write_json(JOB / 'latest-recovery.json', json.loads(checkpoint.with_suffix('.json').read_text()))
        if stop or state['step'] < steps:
            write_json(output / 'interrupted.json', state)
            return 143
        final = WORK / 'models' / stage
        final.parent.mkdir(exist_ok=True)
        export(model, final)
        write_json(output / 'complete.json', dict(state, model=str(final),
                   model_files={p.name: dict(bytes=p.stat().st_size, sha256=file_sha(p))
                                for p in final.iterdir() if p.is_file()}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
