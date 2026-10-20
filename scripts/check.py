"""Run local checks with no network, warehouse access, training CLI, or publication.

Unit tests may fit bounded fixture models. Browser checks are explicit and serve
only docs/ through an isolated loopback server.
"""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from urllib.parse import unquote, urlparse, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def protect_warehouse(event, args):
    if event not in {'open', 'sqlite3.connect'} or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    value = os.fsdecode(args[0])
    if value.startswith('file:'):
        value = unquote(urlparse(value).path)
    path = Path(value).resolve()
    protected = ROOT / 'data/football.sqlite3'
    if path in {protected, Path(str(protected) + '-wal'), Path(str(protected) + '-shm')}:
        raise RuntimeError('Local checks refuse access to the production football warehouse')


class PageLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if 'id' in data:
            if data['id'] in self.ids:
                raise ValueError(f'Duplicate HTML id: {data["id"]}')
            self.ids.add(data['id'])
        for key in ('href', 'src'):
            if key in data and not urlsplit(data[key]).scheme:
                self.links.append(data[key])


def markdown_checks():
    paths = list(ROOT.glob('*.md')) + list((ROOT / 'results').glob('*.md'))
    for path in paths:
        # Inline commands and fenced examples do not define documentation links.
        text = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
        for raw in re.findall(r'(?<!!)\[[^]\n]*\]\(([^\s)]+)\)', text):
            url = urlsplit(raw.strip('<>'))
            if url.scheme or url.netloc or not url.path:
                continue
            target = (path.parent / unquote(url.path)).resolve()
            if not target.is_relative_to(ROOT) or not target.exists():
                raise ValueError(f'Broken documentation link: {path.relative_to(ROOT)}: {raw}')
    return len(paths)


def static_checks():
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Node.js is required for JavaScript checks')
    sources = sorted((ROOT / 'docs').glob('*.js')) + sorted((ROOT / 'tests').glob('*.cjs'))
    for path in sources:
        subprocess.run([node, '--check', str(path)], check=True)
    for name in ('run.sh', 'rebuild.sh'):
        subprocess.run(['bash', '-n', str(ROOT / name)], check=True)
    pages = {}
    for path in (ROOT / 'docs').glob('*.html'):
        page = PageLinks()
        page.feed(path.read_text())
        pages[path] = page
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if not target.is_relative_to(ROOT / 'docs') or not target.is_file():
                raise ValueError(f'Broken local page resource: {path.name}: {link}')
            if url.fragment and target in pages and target.name != 'research.html' and url.fragment not in pages[target].ids:
                raise ValueError(f'Broken page anchor: {path.name}: {link}')
    # Parse source without producing cache directories.
    import ast
    for folder in ('app', 'tests', 'scripts'):
        for path in (ROOT / folder).rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
    documents = markdown_checks()
    print(json.dumps({'static_checks': 'passed', 'pages': len(pages), 'javascript': len(sources), 'documents': documents}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tests', nargs='*', help='Optional unittest module or test names; default discovers tests/.')
    parser.add_argument('--static-only', action='store_true', help='Skip Python runtime tests; still verify sources and manifest.')
    parser.add_argument('--browser', action='store_true', help='Run isolated Playwright checks; missing runtime is a failure.')
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.addaudithook(protect_warehouse)
    if not args.static_only:
        suite = unittest.defaultTestLoader.loadTestsFromNames(args.tests) if args.tests else unittest.defaultTestLoader.discover('tests')
        if suite.countTestCases() == 0:
            raise RuntimeError('No tests collected')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            return 1
    static_checks()
    from scripts.public_manifest import inspect
    print(json.dumps(inspect()))
    if args.browser:
        subprocess.run(['node', str(ROOT / 'tests/frontend_browser.cjs')], check=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'Checks incomplete: {error}', file=sys.stderr)
        raise SystemExit(1)
