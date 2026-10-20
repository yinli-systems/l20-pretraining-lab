"""CPU exclusion index for all normalized 13-word spans in prior prompts.

64-bit hash collisions can only exclude extra text, not admit a matching span.
This index is a conservative overlap screen, not semantic novelty proof.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import numpy as np

WORK = Path('/dev/shm/video-model-validation48')
JOB = Path('/workspace/video-model/validation-48h')


def spans(text):
    words = text.split()
    return [int.from_bytes(hashlib.blake2b(' '.join(words[i:i+13]).encode(), digest_size=8).digest(), 'little')
            for i in range(max(0, len(words)-12))]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--deadline', type=float, required=True); args = p.parse_args()
    marker = JOB/'prior-ngram-complete.json'
    if marker.exists():
        print(marker.read_text()); return
    source = WORK/'prior-prompts.sqlite'
    db = sqlite3.connect('file:'+str(source)+'?mode=ro', uri=True)
    temp = WORK/'prior-13word-hashes.partial.u64'
    count = 0; total = 0
    with temp.open('wb') as out:
        for (text,) in db.execute('SELECT normalized FROM prompts ORDER BY rowid'):
            if time.time() > args.deadline:
                raise TimeoutError('Preparation deadline while indexing prior spans')
            values = spans(text)
            array('Q', values).tofile(out); count += 1; total += len(values)
            if count % 50000 == 0:
                print(json.dumps(dict(prior_prompts_indexed=count, spans=total)), flush=True)
    db.close()
    values = np.unique(np.fromfile(temp, dtype=np.uint64))
    output = WORK/'prior-13word-hashes.npy'
    np.save(output, values)
    with output.open('rb') as stream: sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    receipt = dict(status='complete', prompts=count, total_spans=total, distinct_hashes=len(values),
                   path=str(output), bytes=output.stat().st_size, sha256=sha,
                   source_sha256=json.loads((JOB/'prior-prompt-index.json').read_text())['sha256'],
                   algorithm='all normalized contiguous13word spans, BLAKE2b64, sorted unique',
                   limitations=['Hash collisions only cause conservative over-exclusion.',
                                'No claim of exhaustive paraphrase or pretraining-exposure detection.'])
    marker.write_text(json.dumps(receipt, indent=2)+'\n'); temp.unlink()
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__': main()
