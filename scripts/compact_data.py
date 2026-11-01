"""Losslessly gzip verified source JSON/JSONL and receipt-backed JSON caches.

Plan by default. --apply replaces each plain representation only after complete
round-trip verification; source inventories and logical SHA-256 remain unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.file_io import write_json_atomic


def _regular(path):
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError(f'Symlink is not a compaction source: {path}')
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f'Compaction requires one regular, unshared file: {path}')
    return info


def _sync_directory(path):
    # Unlike ordinary receipts, retiring an original requires durable publication.
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _digest(handle):
    digest, size = hashlib.sha256(), 0
    for block in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(block); size += len(block)
    return digest.hexdigest(), size


def candidates(source_roots, caches, minimum_bytes):
    """Use original inventories, never infer source identity from compressed bytes."""
    rows, protected = {}, set()
    for directory in source_roots:
        directory = Path(directory).absolute()
        manifest = directory / 'manifest.json'
        _regular(manifest)
        payload = json.loads(manifest.read_text())
        if payload.get('complete') is not True or payload.get('dataset') not in ('statsbomb', 'skillcorner'):
            raise ValueError('Only complete StatsBomb/SkillCorner inventories are supported')
        protected.add(manifest)
        original_paths = {entry['path'] for entry in payload['files']}
        for entry in payload['files']:
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Source inventory path escapes root')
            if (relative.suffix not in ('.json', '.jsonl')
                    or relative.as_posix()+'.gz' in original_paths):
                continue  # A native gzip sibling is a separate source asset.
            size, digest = entry['bytes'], entry['sha256']
            if type(size) is not int or size < 0 or not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest):
                raise ValueError('Invalid original source identity')
            if size >= minimum_bytes:
                path = directory / relative
                row = {'path': str(path), 'sha256': digest, 'bytes': size, 'kind': 'source'}
                if path in rows and rows[path] != row:
                    raise ValueError('Conflicting source inventories')
                rows[path] = row
    for name in caches:
        path = Path(name).absolute()
        if path.suffix != '.json':
            raise ValueError('Only receipt-backed JSON caches are supported')
        receipt = path.with_suffix('.receipt.json')
        protected.update((path, Path(str(path)+'.gz'), receipt))
        _regular(receipt)
        digest = json.loads(receipt.read_text())['content_sha256']
        if not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest):
            raise ValueError('Invalid cache content identity')
        physical = path if path.exists() else Path(str(path)+'.gz')
        info = _regular(physical)
        if physical == path and info.st_size < minimum_bytes:
            continue
        if path in rows:
            raise ValueError('Cache overlaps a source inventory')
        rows[path] = {'path': str(path), 'sha256': digest,
                      'bytes': info.st_size if physical == path else None, 'kind': 'cache'}
        protected.add(receipt)
    return list(rows.values()), protected


def compact_file(entry, *, level=3):
    path = Path(entry['path']); target = Path(str(path)+'.gz')
    expected = entry['sha256']; expected_size = entry['bytes']
    row = dict(entry, stored_path=str(target), state='already_compressed', reclaimed_allocated_bytes=0)
    if not path.exists():
        _regular(target)
        with gzip.open(target, 'rb') as handle:
            digest, size = _digest(handle)
        if digest != expected or expected_size is not None and size != expected_size:
            raise ValueError('Existing compressed representation does not match original identity')
        return {**row, 'bytes': size, 'stored_bytes': target.stat().st_size}
    initial = _regular(path)
    if expected_size is not None and initial.st_size != expected_size:
        raise ValueError('Plain source size no longer matches inventory')
    temporary = None
    created = not target.exists()
    try:
        if created:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.'+path.name+'.compact-', suffix='.part', delete=False) as output:
                temporary = Path(output.name)
                with gzip.GzipFile(filename='', mode='wb', fileobj=output, compresslevel=level, mtime=0) as compressed, path.open('rb') as source:
                    digest, size = hashlib.sha256(), 0
                    for block in iter(lambda: source.read(1024*1024), b''):
                        compressed.write(block); digest.update(block); size += len(block)
                if digest.hexdigest() != expected or size != initial.st_size:
                    raise ValueError('Plain source checksum changed; keeping original')
                os.fchmod(output.fileno(), stat.S_IMODE(initial.st_mode))
                output.flush(); os.fsync(output.fileno())
            candidate = temporary
        else:
            _regular(target)
            with path.open('rb') as source:
                digest, size = _digest(source)
            if digest != expected or size != initial.st_size:
                raise ValueError('Plain source checksum changed; keeping original')
            candidate = target
        with gzip.open(candidate, 'rb') as handle:
            digest, size = _digest(handle)
        if digest != expected or size != initial.st_size:
            raise ValueError('Compressed round trip failed; keeping original')
        if created:
            os.utime(candidate, ns=(initial.st_atime_ns, initial.st_mtime_ns))
        stored = candidate.stat()
        if stored.st_size >= initial.st_size:
            return {**row, 'state': 'not_beneficial', 'stored_path': str(path),
                    'bytes': initial.st_size, 'stored_bytes': initial.st_size}
        if _identity(_regular(path)) != _identity(initial):
            raise ValueError('Source changed during compression; keeping both representations')
        if created:
            # Publish without overwriting a concurrently created/previous representation.
            os.link(candidate, target)
            temporary.unlink(); temporary = None
        # Also sync a pre-existing compressed copy before removing the original.
        with target.open('rb') as durable:
            os.fsync(durable.fileno())
        _sync_directory(path.parent)
        if _identity(_regular(path)) != _identity(initial):
            raise ValueError('Source changed before retirement; keeping both representations')
        path.unlink()
        _sync_directory(path.parent)
        row.update(original_mtime_ns=initial.st_mtime_ns, original_mode=stat.S_IMODE(initial.st_mode))
        freed = initial.st_blocks*512 - (stored.st_blocks*512 if created else 0)
        return {**row, 'state': 'compressed' if created else 'duplicate_plain_retired',
                'bytes': size, 'stored_bytes': stored.st_size, 'reclaimed_allocated_bytes': freed}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _open_inputs(paths):
    """Refuse visible active users; this is scoped to the current /proc namespace."""
    roots = tuple(str(Path(p).resolve()) for p in paths)
    users = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            for descriptor in (process/'fd').iterdir():
                try:
                    value = os.readlink(descriptor)
                except FileNotFoundError:
                    continue
                if any(value == root or value.startswith(root+'/') for root in roots):
                    users.append({'pid': process.name, 'path': value})
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise RuntimeError('Cannot inspect open-file users; refusing to apply') from exc
    if users:
        raise RuntimeError(f'Sources have visible open-file users: {users[:8]}')


def run(args):
    if args.minimum_bytes < 0 or not 1 <= args.level <= 9 or not 1 <= args.workers <= 4:
        raise ValueError('Invalid minimum size, gzip level or workers')
    rows, protected = candidates(args.source_root, args.cache, args.minimum_bytes)
    output = args.report.resolve()
    for source in args.source_root:
        if output.is_relative_to(Path(source).resolve()):
            raise ValueError('Report cannot overwrite a source tree')
    if output in {p.resolve() for p in protected} or any(
            output in (Path(r['path']).resolve(), Path(r['path']+'.gz').resolve()) for r in rows):
        raise ValueError('Report cannot overwrite an input or its identity receipt')
    if output.exists() and (not output.is_file() or json.loads(output.read_text()).get('schema') != 'SoccerAnalytics.DataCompaction.v1'):
        raise FileExistsError('Refusing to overwrite an unrelated file with a storage receipt')
    if output.suffix != '.json' or any(output.is_relative_to(ROOT/name) for name in ('docs','results','data','models')):
        raise ValueError('Storage receipt requires a private JSON output, e.g. .tmp/data-compaction/report.json')
    report = {'schema': 'SoccerAnalytics.DataCompaction.v1', 'state': 'planned',
              'scope': 'Lossless gzip of original logical bytes; no SQLite rewrite, schema change or source manifest edits.',
              'level': args.level, 'workers': args.workers, 'minimum_bytes': args.minimum_bytes,
              'selected_files': len(rows), 'selected_known_logical_bytes': sum(r['bytes'] or 0 for r in rows),
              'unknown_logical_size_files': sum(r['bytes'] is None for r in rows),
              'results': []}
    if not args.apply:
        report['candidates'] = rows
        write_json_atomic(output, report)
        return report
    _open_inputs([*args.source_root, *args.cache])
    started, last_saved = time.monotonic(), 0.
    iterator = iter(rows)
    report['state'] = 'running'
    write_json_atomic(output, report)
    def record(future, entry):
        try:
            report['results'].append(future.result())
        except Exception as exc:
            report['results'].append({**entry, 'state': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
    pending = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        def fill():
            for entry in iterator:
                pending[pool.submit(compact_file, entry, level=args.level)] = entry
                if len(pending) == args.workers:
                    break
        fill()
        try:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    record(future, pending.pop(future))
                fill()
                now = time.monotonic()
                if now-last_saved >= 10:
                    report['elapsed_seconds'] = now-started
                    write_json_atomic(output, report)
                    print(json.dumps({'completed':len(report['results']),'selected':len(rows),
                        'reclaimed_allocated_bytes':sum(r.get('reclaimed_allocated_bytes',0) for r in report['results'])}),flush=True)
                    last_saved=now
        except BaseException:
            for future,entry in pending.items():
                record(future,entry)
            report['state']='interrupted'; write_json_atomic(output,report)
            raise
    report.update(state='partial' if any(r['state']=='failed' for r in report['results']) else 'complete',
                  elapsed_seconds=time.monotonic()-started,
                  reclaimed_allocated_bytes=sum(r.get('reclaimed_allocated_bytes',0) for r in report['results']))
    write_json_atomic(output,report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', action='append',type=Path,default=[])
    parser.add_argument('--cache',action='append',type=Path,default=[])
    parser.add_argument('--minimum-bytes',type=int,default=256*1024)
    parser.add_argument('--level',type=int,default=3)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args(argv)
    if not args.source_root and not args.cache:
        parser.error('Supply a supported source root or a receipt-backed cache')
    report=run(args)
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','candidates')},ensure_ascii=False))
    return 1 if report['state']=='partial' else 0


if __name__=='__main__':
    raise SystemExit(main())
