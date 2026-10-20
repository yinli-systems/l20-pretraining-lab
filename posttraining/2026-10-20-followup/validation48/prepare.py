"""CPU-only, resumable provenance/access preflight for the authorized pilots.

This deliberately never marks training ready merely because a dataset downloads.
Full data/label audits and a frozen new evaluation suite are separate gates.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import collections
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import time

OLD = Path('/workspace/video-model')
JOB = OLD / 'validation-48h'
WORK = Path('/dev/shm/video-model-validation48')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def status(phase, **extra):
    row = dict(phase=phase, utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               pid=os.getpid(), gpu_training_started=False, **extra)
    write(JOB / 'preparation-status.json', row)
    print(json.dumps(row), flush=True)


def normalize(text):
    return ' '.join(re.findall(r'\w+', text.lower()))


def family(text):
    # This conservative numeric abstraction is one exclusion key, not proof of
    # semantic independence. Template/source family identifiers remain required.
    return re.sub(r'\d+', '#', normalize(text))


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def old_index(deadline):
    import numpy as np
    sys.path.insert(0, str(OLD))
    from common import tokenizer
    tok = tokenizer()
    WORK.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(WORK / 'prior-prompts.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS prompts(
          hash TEXT PRIMARY KEY, family_hash TEXT NOT NULL, normalized TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS cursors(source TEXT PRIMARY KEY, n INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS families ON prompts(family_hash);
    ''')
    total = 0
    sources = []
    for p in sorted(Path('/dev/shm/l20-posttrain/prepared').iterdir()):
        if not (p / 'complete').exists():
            continue
        off = np.load(p / 'offsets.npy')
        ids = np.memmap(p / 'ids.u16', dtype=np.uint16, mode='r')
        flags = np.load(p / 'validation.npy')
        start = db.execute('SELECT n FROM cursors WHERE source=?', (p.name,)).fetchone()
        start = start[0] if start else 0
        n = len(off) - 1
        for begin in range(start, n, 512):
            if time.time() >= deadline:
                db.close()
                raise TimeoutError('Preparation time bound reached while indexing')
            end = min(n, begin + 512)
            texts = tok.batch_decode([ids[int(off[i]):int(off[i+1])].tolist()
                                     for i in range(begin, end)], skip_special_tokens=False)
            for i, text in zip(range(begin, end), texts):
                # Index train and prior validation conservatively. Excluding
                # both avoids repurposing previously inspected validation items.
                prompts = re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', text, re.S)
                if not prompts:
                    raise RuntimeError(f'Cannot reconstruct prompt: {p.name}:{i}')
                prompt = '\n'.join(prompts)
                norm = normalize(prompt)
                db.execute('INSERT OR IGNORE INTO prompts VALUES(?,?,?)',
                           (digest(norm), digest(family(prompt)), norm))
            db.execute('INSERT OR REPLACE INTO cursors VALUES(?,?)', (p.name, end))
            db.commit()
            if begin % 8192 == 0:
                status('indexing_previous_prompts', source=p.name, indexed=end, source_total=n)
        sources.append(dict(source=p.name, records=n, prior_validation_records=int(flags.sum())))
        total += n
    # Include the exact original evaluation and sampled RFT pools as exclusions.
    extras = [OLD / 'benchmarks.jsonl']
    extras += list(Path('/dev/shm/l20-posttrain').glob('rft-*/accepted.jsonl'))
    for path in extras:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            text = row.get('prompt', '')
            if text:
                db.execute('INSERT OR IGNORE INTO prompts VALUES(?,?,?)',
                           (digest(normalize(text)), digest(family(text)), normalize(text)))
    db.commit()
    count = db.execute('SELECT COUNT(*) FROM prompts').fetchone()[0]
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()
    path = WORK / 'prior-prompts.sqlite'
    sha = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    report = dict(status='complete', prepared_records_indexed=total,
                  distinct_prompt_hashes=count, sources=sources, path=str(path),
                  sha256=sha, bytes=path.stat().st_size,
                  exclusions='all retained old training and validation prompts plus original evaluation',
                  limitations=['Unknown original pretraining exposure cannot be reconstructed.',
                               'Exact and numeric-family keys do not replace source/template grouping or near-duplicate review.'])
    write(JOB / 'prior-prompt-index.json', report)
    return report


SPECS = [
    ('cpt-math', 'HuggingFaceTB/finemath', 'finemath-4plus', 'train'),
    ('cpt-code', 'HuggingFaceTB/stack-edu', 'Python', 'train'),
    ('cpt-general', 'HuggingFaceFW/fineweb-edu', 'sample-10BT', 'train'),
    ('sft-math', 'nvidia/OpenMathInstruct-2', None, 'train_1M'),
    ('sft-code', 'nvidia/OpenCodeInstruct', 'train', 'train'),
    ('sft-general', 'HuggingFaceTB/smoltalk', 'smol-magpie-ultra', 'train'),
    ('word-eval-candidate', 'math-ai/TemplateGSM', 'templategsm-7473-1k', 'train'),
    ('competition-eval-candidate', 'SynthLabsAI/Big-Math-RL-Verified', None, 'train'),
    ('retention-eval-candidate', 'TIGER-Lab/MMLU-Pro', None, 'test'),
]


def access_audit(deadline):
    from huggingface_hub import HfApi, hf_hub_download
    from datasets import load_dataset
    import requests
    api = HfApi()
    folder = JOB / 'source-access'
    folder.mkdir(exist_ok=True)
    results = []
    for key, repo, config, split in SPECS:
        if time.time() > deadline - 120:
            raise TimeoutError('Preparation bound reached during source access checks')
        report_path = folder / (key + '.json')
        if report_path.exists():
            results.append(json.loads(report_path.read_text()))
            continue
        row = dict(key=key, repo=repo, config=config, split=split, ready_for_training=False)
        try:
            info = api.dataset_info(repo, timeout=30)
            row['revision'] = info.sha
            card = info.card_data.to_dict() if info.card_data else {}
            row['license_metadata'] = card.get('license')
            ds = load_dataset(repo, config, split=split, revision=info.sha, streaming=True)
            examples = []
            for r in ds:
                examples.append(r)
                if len(examples) == 8:
                    break
            row['columns'] = list(examples[0]) if examples else []
            row['sample_records_received'] = len(examples)
            row['actual_text_present'] = bool(examples and any(
                isinstance(examples[0].get(k), str) and len(examples[0][k]) > 50
                for k in ['text', 'content', 'problem', 'input', 'question']))
            row['status'] = 'schema_and_sample_received'
            # Small samples are private preparation material, not a ready corpus.
            write(folder / (key + '.samples.json'), examples)
            if key == 'cpt-code':
                row['actual_text_present'] = False
                sample = examples[0]
                row['content_access_probe'] = {}
                ident = sample.get('content_id') or sample.get('blob_id')
                if ident:
                    url = f'https://softwareheritage.s3.amazonaws.com/content/{ident}'
                    response = requests.get(url, timeout=20, stream=True)
                    row['content_access_probe'] = dict(status=response.status_code,
                                                       endpoint='public unsigned SWH S3')
                    if response.status_code == 200:
                        import gzip
                        raw = response.raw.read(1024 * 1024)
                        text = gzip.decompress(raw).decode(sample.get('src_encoding') or 'utf-8')
                        row['actual_text_present'] = bool(text)
                        row['content_access_probe']['bytes_received'] = len(raw)
                    response.close()
                row['status'] = 'content_access_verified' if row['actual_text_present'] else 'metadata_only_content_access_unresolved'
        except Exception as exc:
            row.update(status='access_failed', error=str(exc)[:500])
        write(report_path, row)
        results.append(row)
        status('checking_source_access', source=key, result=row['status'])
    write(JOB / 'source-access.json', results)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--deadline', type=float, required=True)
    args = p.parse_args()
    JOB.mkdir(parents=True, exist_ok=True)
    lock = (JOB / 'prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status('started_cpu_preparation', deadline_utc=dt.datetime.fromtimestamp(args.deadline, dt.timezone.utc).isoformat())
    try:
        access = access_audit(args.deadline)
        prior = old_index(args.deadline)
        write(JOB / 'readiness.json', dict(ready=False,
              completed_gates=['previous_prompt_index', 'pinned_source_access_audit'],
              pending_gates=['audited new SFT corpus', 'CPT code content and measured mixture',
                             'frozen reviewed 1200-dev/2400-confirmation and retention suite',
                             'native objective and gradient parity', 'GPU throughput preflight',
                             'storage-aware recovery and verified backup capacity'],
              prior_prompt_index=prior, source_access_summary={r['key']:r['status'] for r in access}))
        status('initial_preparation_complete', training_ready=False,
               next_action='Review content access and curate verified corpora/evaluation before starting GPU budget')
    except Exception as exc:
        status('preparation_failed', error=repr(exc), training_ready=False)
        raise


if __name__ == '__main__':
    main()
