"""Explicit native-2048 CPT/SFT labels and deterministic resumable sampling."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

CONTEXT = 2048


def file_sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def make_labels(ids, mask, objective):
    if objective not in ('cpt', 'sft'):
        raise ValueError('Explicit cpt or sft objective required')
    if ids.shape != mask.shape:
        raise ValueError('Input and supervision mask dimensions differ')
    # For CPT the recorded mask must describe all nonpadding content including
    # real endoftext separators. Token value alone cannot distinguish padding
    # from the real endoftext token (both can be 0 for this tokenizer).
    labels = ids.clone()
    labels[~mask.bool()] = -100
    return labels


class Packed(Dataset):
    def __init__(self, directory, objective):
        self.path = Path(directory)
        self.manifest_path = self.path / 'manifest.json'
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest['objective'] != objective:
            raise ValueError('Data manifest objective mismatch')
        if self.manifest['context'] != CONTEXT:
            raise ValueError('Context extension is outside this experiment')
        if not self.manifest.get('quality_audit_passed'):
            raise ValueError('Training data has not passed quality gates')
        for name in ['train.ids.u16', 'train.mask.u8']:
            expected = self.manifest['files'][name]
            path = self.path / name
            if path.stat().st_size != expected['bytes'] or file_sha(path) != expected['sha256']:
                raise ValueError('Packed data integrity mismatch: ' + name)
        self.ids = np.memmap(self.path / 'train.ids.u16', mode='r', dtype=np.uint16).reshape(-1, CONTEXT)
        self.mask = np.memmap(self.path / 'train.mask.u8', mode='r', dtype=np.uint8).reshape(-1, CONTEXT)
        if self.ids.shape != self.mask.shape or not len(self.ids):
            raise ValueError('Empty or mismatched packed data')
        if self.mask.max() > 1 or (self.mask[:, 1:].sum(1) == 0).any():
            raise ValueError('Invalid or empty shifted supervision mask')
        self.objective = objective
        self.identity = file_sha(self.manifest_path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        ids = torch.tensor(np.array(self.ids[i], dtype=np.int64))
        mask = torch.tensor(np.array(self.mask[i], dtype=np.bool_))
        return ids, make_labels(ids, mask, self.objective), int(mask[1:].sum())


class ResumableBatches:
    """Cursor records batches consumed by the optimizer, never prefetched ones."""
    def __init__(self, n, batch, seed, consumed=0):
        if n < batch:
            raise ValueError('Dataset smaller than microbatch')
        self.n, self.batch, self.seed, self.consumed = n, batch, seed, consumed
        self.per_epoch = n // batch

    def __len__(self):
        return self.per_epoch

    def __iter__(self):
        epoch, offset = divmod(self.consumed, self.per_epoch)
        order = np.random.default_rng(self.seed + epoch).permutation(self.n)
        for j in range(offset, self.per_epoch):
            yield order[j*self.batch:(j+1)*self.batch].tolist()


def optimizer_step_budget(nblocks, microbatch, accumulation, token_cap, max_epochs=None):
    tokens_per_update = microbatch * accumulation * CONTEXT
    token_steps = token_cap // tokens_per_update
    if max_epochs is not None:
        # Conservative complete-update bound; never silently exceed two epochs.
        epoch_steps = (nblocks // microbatch) * max_epochs // accumulation
        token_steps = min(token_steps, int(epoch_steps))
    if token_steps < 1:
        raise ValueError('Insufficient data/token budget for one optimizer update')
    return int(token_steps)
