"""Verify or refresh only explicitly approved public project files.

A refresh adds eligible source/docs and selected public results. It never scans
private data, experiment caches, models, environments, or browser evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'results/public-manifest.json'
PRIVATE_ROOTS = {'.git', '.tmp', '.venv', 'data', 'models', 'output', 'archive', 'matches', 'log'}
PRIVATE_SUFFIXES = {'.sqlite3', '.db', '.bundle', '.npz', '.npy', '.pkl', '.pickle', '.joblib', '.pyc'}
SOURCE_SUFFIXES = {'.py', '.js', '.cjs', '.css', '.html', '.yml', '.yaml', '.md', '.txt', '.sh'}


def approved_file(relative):
    """Validate a manifest path before stat/open, including every symlink ancestor."""
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] in PRIVATE_ROOTS:
        raise ValueError(f'Private or unsafe public path: {relative}')
    if path.suffix in PRIVATE_SUFFIXES or any(part in {'__pycache__', 'node_modules'} for part in path.parts):
        raise ValueError(f'Private artifact cannot be published: {relative}')
    current = ROOT
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'Symlink cannot be published: {relative}')
    if not current.resolve().is_relative_to(ROOT):
        raise ValueError(f'Public path escapes repository: {relative}')
    return current


def source_files():
    files = set()
    for folder in ('app', 'tests', 'scripts', 'docs', '.github/workflows'):
        for path in (ROOT / folder).rglob('*'):
            relative = path.relative_to(ROOT)
            if '__pycache__' not in relative.parts and path.suffix in SOURCE_SUFFIXES:
                if approved_file(relative).is_file():
                    files.add(relative.as_posix())
    for path in ROOT.iterdir():
        if path.is_file() and (path.suffix in SOURCE_SUFFIXES or path.name in {'.gitignore', '.dockerignore', 'Dockerfile'}):
            files.add(path.name)
    return files


def inspect(*, refresh=False):
    manifest = approved_file(MANIFEST.relative_to(ROOT))
    data = json.loads(manifest.read_text())
    entries = data['files']
    names = [entry['path'] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate manifest paths')
    for name in names:
        approved_file(name)
    if refresh:
        names = sorted({name for name in names if approved_file(name).is_file()} | source_files())
    rows = []
    for name in names:
        path = approved_file(name)
        raw = path.read_bytes()
        if path.suffix == '.json':
            json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        rows.append({'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    if not refresh and (missing := source_files() - set(names)):
        raise ValueError('Source files missing from manifest: ' + ', '.join(sorted(missing)))
    if not refresh and rows != entries:
        original = {row['path']: row for row in entries}
        changed = [row['path'] for row in rows if row != original.get(row['path'])]
        raise ValueError('Public files changed: ' + ', '.join(changed))
    gallery = json.loads(approved_file('docs/data/research/figures.json').read_text())['figures']
    for figure in gallery:
        name = 'docs/data/research/figures/' + figure['filename']
        if name not in names:
            raise ValueError(f'Gallery asset is not in the manifest: {name}')
        approved_file(name)
    if refresh:
        from app.file_io import write_json_atomic
        data['files'] = rows
        write_json_atomic(manifest, data)
    return {'public_files': len(rows), 'gallery_assets': len(gallery), 'status': 'refreshed' if refresh else 'verified'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', help='Update approved source file identities; no data discovery.')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(inspect(refresh=args.refresh)))
    except (OSError, KeyError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
