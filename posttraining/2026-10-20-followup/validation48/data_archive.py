"""Lossless frozen-data backup through the existing receipt synchronizer.

Build is CPU-only and accepts explicit immutable identities. Verify decompresses
every byte locally before acknowledging anything. This does not approve data
quality, start a monitor, replace a checkpoint, or authorize instance shutdown.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def decoded_identity(path):
    value = hashlib.sha256()
    size = 0
    with gzip.open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
            size += len(block)
    return dict(bytes=size, sha256=value.hexdigest())


def build(plan_path, destination):
    plan_path, destination = Path(plan_path), Path(destination)
    plan = json.loads(plan_path.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for entry in plan['files']:
        source = Path(entry['path'])
        if not source.is_absolute() or source.is_symlink():
            raise ValueError('Use an absolute immutable regular-file source')
        expected = {key: entry[key] for key in ['bytes', 'sha256']}
        if source.stat().st_size != expected['bytes'] or sha(source) != expected['sha256']:
            raise ValueError('Source differs from frozen plan: ' + str(source))
        name = hashlib.sha256(str(source).encode()).hexdigest()[:16] + '-' + source.name + '.gz'
        output = destination / name
        if output.exists():
            if decoded_identity(output) != expected:
                raise ValueError('Existing archive differs; do not overwrite it silently')
        else:
            # gzip can expand incompressible inputs slightly. Reserve the full
            # raw length plus margin before writing, rather than assuming ratio.
            if shutil.disk_usage(destination).free < expected['bytes'] * 1.02 + 256 * 1024**2:
                raise RuntimeError('Insufficient remote archive capacity')
            temporary = output.with_suffix('.gz.tmp')
            digest, size = hashlib.sha256(), 0
            try:
                with source.open('rb') as incoming, temporary.open('wb') as raw:
                    with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=3, mtime=0) as outgoing:
                        for block in iter(lambda: incoming.read(8 * 1024**2), b''):
                            outgoing.write(block)
                            digest.update(block)
                            size += len(block)
                if dict(bytes=size, sha256=digest.hexdigest()) != expected:
                    raise ValueError('Source changed during compression')
                if decoded_identity(temporary) != expected:
                    raise ValueError('Archive decompression failed original-byte identity')
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
        files[name] = dict(source=str(source), role=entry['role'], original=expected,
                           archive=dict(bytes=output.stat().st_size, sha256=sha(output)))
    result = dict(schema_version=1, plan_sha256=sha(plan_path), files=files,
                  raw_bytes=sum(row['original']['bytes'] for row in files.values()),
                  compressed_bytes=sum(row['archive']['bytes'] for row in files.values()),
                  created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  data_quality_approved=False,
                  restore='gunzip each archive to its recorded source path; verify original SHA256 before use')
    write(destination / 'manifest.json', result)
    return result


def verify(directory, marker, future_required_bytes=0, headroom_bytes=1536 * 1024**2):
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    files = {}
    for name, row in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Unexpected archive path')
        path = directory / name
        if path.stat().st_size != row['archive']['bytes'] or sha(path) != row['archive']['sha256']:
            raise ValueError('Transferred archive checksum mismatch: ' + name)
        if decoded_identity(path) != row['original']:
            raise ValueError('Restored original checksum mismatch: ' + name)
        files[name] = dict(path=str(path.resolve()), **row, original_bytes_verified=True)
    free = shutil.disk_usage(directory).free
    result = dict(all_files_verified=True, decompressed_original_bytes_verified=True,
                  manifest_sha256=sha(manifest_path), files=files,
                  verified_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  free_bytes_after_archives=free, future_required_bytes=future_required_bytes,
                  headroom_bytes=headroom_bytes,
                  capacity_passed=free >= future_required_bytes + headroom_bytes,
                  models_and_recovery_are_not_compressed=True, data_quality_approved=False)
    write(marker, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    command = sub.add_parser('build')
    command.add_argument('--plan', required=True)
    command.add_argument('--destination', required=True)
    command = sub.add_parser('verify')
    command.add_argument('--directory', required=True)
    command.add_argument('--marker', required=True)
    command.add_argument('--future-required-bytes', type=int, default=0)
    args = parser.parse_args()
    value = (build(args.plan, args.destination) if args.action == 'build' else
             verify(args.directory, args.marker, args.future_required_bytes))
    print(json.dumps({key: value[key] for key in value if key != 'files'}, indent=2))


if __name__ == '__main__':
    main()
