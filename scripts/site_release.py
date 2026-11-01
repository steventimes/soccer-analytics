"""Package a clean, fixed docs commit with file identities; never deploy it."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def build_site_release(repo, output, *, commit, workflow_commit, run_id, run_attempt):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if (not re.fullmatch(r'[a-f0-9]{40}', commit)
            or not re.fullmatch(r'[a-f0-9]{40}', workflow_commit)
            or not str(run_id).isdigit() or not str(run_attempt).isdigit()):
        raise ValueError('Release requires full source/workflow commits and numeric workflow identity')
    if commit != workflow_commit:
        raise ValueError('Source commit does not match the validation workflow commit')
    if output.exists() or output.is_relative_to(repo):
        raise ValueError('Release output must be a new directory outside the checkout')

    def git(*args):
        return subprocess.run(['git', *args], cwd=repo, check=True,
                              capture_output=True, timeout=30).stdout

    if git('rev-parse', 'HEAD').decode().strip() != commit:
        raise ValueError('Checkout does not match the validated commit')
    git('diff', '--exit-code', 'HEAD', '--', 'docs')
    tree = git('ls-tree', '-r', '-z', commit, '--', 'docs')
    entries = [row.split(b'\t', 1) for row in tree.split(b'\0') if row]
    if not entries:
        raise ValueError('Release has no tracked docs files')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.site-release-', dir=output.parent) as temporary:
        staging = Path(temporary) / 'site'
        staging.mkdir()
        files = []
        for metadata, raw_name in entries:
            mode, kind, _ = metadata.decode().split()
            name = Path(os.fsdecode(raw_name))
            source = repo / name
            if mode not in {'100644', '100755'} or kind != 'blob' or source.is_symlink():
                raise ValueError('Release only supports regular tracked docs files')
            relative = name.relative_to('docs')
            if relative.as_posix() == 'site-release.json':
                raise ValueError('Release identity is generated during packaging')
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            raw = destination.read_bytes()
            files.append({'path': relative.as_posix(), 'bytes': len(raw),
                          'sha256': hashlib.sha256(raw).hexdigest()})
        if not {'index.html', 'data/manifest.json'} <= {row['path'] for row in files}:
            raise ValueError('Release lacks the application or prediction manifest')
        files.sort(key=lambda row: row['path'])
        identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        receipt = {'schema': 'Touchline.SiteRelease.v1', 'source_commit': commit,
                   'validation_workflow_commit': workflow_commit,
                   'workflow_run_id': str(run_id), 'workflow_run_attempt': str(run_attempt),
                   'validation_workflow': 'Project checks', 'files_sha256': identity,
                   'files': files, 'limits': ['File identities do not attest live data freshness or forecast accuracy.']}
        (staging / 'site-release.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if output.exists():
            raise ValueError('Release output appeared during packaging')
        os.replace(staging, output)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--workflow-commit', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--run-attempt', required=True)
    args = parser.parse_args(argv)
    receipt = build_site_release(args.repo, args.output, commit=args.commit,
                                 workflow_commit=args.workflow_commit,
                                 run_id=args.run_id, run_attempt=args.run_attempt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'files'}))


if __name__ == '__main__':
    main()
