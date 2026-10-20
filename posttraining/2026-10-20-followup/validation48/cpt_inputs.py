"""Bind each original or top-up CPT segment to a frozen completion receipt."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def checked_segments(value, domain, job):
    legacy = isinstance(value, str)
    entries = [dict(root=value, completion=str(job / f'cpt-{domain}-complete.json'))] if legacy else value
    if not isinstance(entries, list) or not entries:
        raise ValueError('CPT domain must declare at least one immutable segment')
    result, roots, source_spec = [], set(), None
    for entry in entries:
        root = Path(entry['root'])
        completion_path = Path(entry['completion'])
        if not root.is_absolute() or not completion_path.is_absolute():
            raise ValueError('CPT segment requires absolute paths')
        digest = sha(completion_path)
        if not legacy and digest != entry['completion_sha256']:
            raise ValueError('CPT completion receipt changed')
        complete = json.loads(completion_path.read_text())
        if Path(complete['path']).resolve() != root.resolve() or root.resolve() in roots:
            raise ValueError('CPT root mismatches completion or is repeated')
        roots.add(root.resolve())
        if complete['domain'] != domain:
            raise ValueError('Wrong CPT domain')
        if source_spec is None:
            source_spec = complete['source']
        elif complete['source'] != source_spec:
            raise ValueError('Top-up cannot silently substitute the pinned source')
        identity = {str(completion_path): digest}
        for name in ['ids.u16', 'documents.jsonl', 'offsets.npy']:
            expected = complete['files'][name]
            path = root / name
            if path.is_symlink() or path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
                raise ValueError('CPT segment content identity changed')
            identity[str(path)] = expected['sha256']
        result.append((root, complete, identity))
    # A top-up's skipped prefix must identify the same original completion,
    # rather than a receipt from a different acquisition or an altered cursor.
    for _, complete, _ in result:
        parent = complete.get('parent_complete_sha256')
        if parent and parent not in {sha(Path(entry['completion'])) for entry in entries}:
            raise ValueError('Top-up parent completion is absent from the declared segments')
    return result
