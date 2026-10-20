import importlib
import json
import os
import re
import sqlite3
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from app.analytics.artifacts import file_digest
from app.cli.research import analytics_lab
from app.data_service.fetch.public import FootballCSVClient
from app.data_service.warehouse import Warehouse


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "app"


class TestAnalyticsCLI(unittest.TestCase):
    MODULES = {
        "passing-lab": "app.analytics.passing_context",
        "space-lab": "app.analytics.receiving_space",
        "value-lab": "app.analytics.action_value",
        "buildup-lab": "app.analytics.buildup_roles",
    }

    def setup_run(self, root, command, *, missing=None, manifest_change=None):
        output = root / "output"
        output.mkdir()
        database = root / "archive.sqlite3"
        with Warehouse(database):
            pass
        args = SimpleNamespace(
            command=command, database=database, output_dir=output,
            data_dir=root / "sources", samples=2, seeds="11,29",
            target_league="England", train_cap=2, epochs=1, horizon=10,
            minimum_minutes=900,
        )
        report = {"run_id": "current-run"}
        if command == "passing-lab":
            names = ["report.json", "provenance.json", "pass-contexts.json", "pressing-chains.json"]
            report["artifacts"] = names
        elif command == "space-lab":
            report["artifacts"] = {
                "provenance": "provenance.json", "numeric_surfaces": "surfaces.npz",
                "private_frames": "frame-context.jsonl",
                "private_clock_assignments": "clock-assignments.json",
                "private_frame_measurements": "frame-measurements.json",
            }
            names = ["report.json", *report["artifacts"].values()]
        elif command == "buildup-lab":
            names = ["report.json", "provenance.json", "predictions.json", "model-weights.json",
                     "understat-panel.json", "current-profiles.json", "wyscout-panels.json", "neighbors.json"]
        else:
            names = ["report.json", "provenance.json", *[f"numeric-{index}.npz" for index in range(42)]]

        def run(*unused_args, **unused_kwargs):
            for name in names:
                if name == missing:
                    continue
                file = output / name
                file.write_bytes(json.dumps(report).encode() if name == "report.json" else b"{}\n")
            if command == "value-lab":
                manifest = {
                    "run_id": report["run_id"],
                    "files": {name: file_digest(output / name) for name in names if name != missing},
                }
                if missing:
                    manifest["files"][missing] = "0" * 64
                if manifest_change:
                    manifest_change(manifest)
                (output / "artifact-manifest.json").write_text(json.dumps(manifest))
            return report

        module = ModuleType(self.MODULES[command])
        module.run = Mock(side_effect=run)
        return args, module, names + (["artifact-manifest.json"] if command == "value-lab" else [])

    def invoke(self, args, module):
        # Stub only the lazily selected experiment; exercise the actual CLI and archive.
        with patch.dict(sys.modules, {self.MODULES[args.command]: module}), redirect_stdout(StringIO()):
            return analytics_lab(args)

    def assert_links(self, args, names):
        with sqlite3.connect(args.database) as db:
            rows = db.execute(
                "SELECT r.run_kind,r.run_id,a.path FROM analysis_run_assets r "
                "JOIN assets a ON a.id=r.asset_id"
            ).fetchall()
            self.assertEqual(set(rows), {(args.command, "current-run", name) for name in names})
            self.assertEqual(db.execute("SELECT COUNT(*) FROM assets").fetchone()[0], len(names))

    def assert_empty_archive(self, args):
        with sqlite3.connect(args.database) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM datasets").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM assets").fetchone()[0], 0)
            table = db.execute("SELECT name FROM sqlite_master WHERE name='analysis_run_assets'").fetchone()
            if table:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM analysis_run_assets").fetchone()[0], 0)

    def test_other_labs_archive_only_current_outputs_and_ignore_stale_manifest(self):
        for command in ("passing-lab", "space-lab", "buildup-lab"):
            for stale_manifest in (False, True):
                with self.subTest(command=command, stale_manifest=stale_manifest), tempfile.TemporaryDirectory() as directory:
                    args, module, names = self.setup_run(Path(directory), command)
                    stale = {"stale-model.npz": b"previous model bytes", "stale-report.json": b'{"old":true}'}
                    if stale_manifest:
                        stale["artifact-manifest.json"] = b'{"run_id":"old-value-run","files":{"missing-old.npz":"bad"}}'
                    for name, content in stale.items():
                        (args.output_dir / name).write_bytes(content)
                    self.assertEqual(self.invoke(args, module), 0)
                    self.assert_links(args, names)
                    for name, content in stale.items():
                        self.assertEqual((args.output_dir / name).read_bytes(), content)

    def test_missing_intended_output_fails_without_partial_archive(self):
        for command in self.MODULES:
            with self.subTest(command=command), tempfile.TemporaryDirectory() as directory:
                args, module, _ = self.setup_run(Path(directory), command, missing="provenance.json")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.invoke(args, module)
                self.assert_empty_archive(args)

    def test_value_archives_verified_current_manifest_only(self):
        with tempfile.TemporaryDirectory() as directory:
            args, module, names = self.setup_run(Path(directory), "value-lab")
            stale = {"stale-model.npz": b"old model", "stale-report.json": b'{"run_id":"old"}'}
            for name, content in stale.items():
                (args.output_dir / name).write_bytes(content)
            self.assertEqual(self.invoke(args, module), 0)
            self.assertEqual(len(names), 45)
            self.assert_links(args, names)
            for name, content in stale.items():
                self.assertEqual((args.output_dir / name).read_bytes(), content)

    def test_value_requires_current_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            args, module, _ = self.setup_run(Path(directory), "value-lab")
            produce = module.run.side_effect

            def without_manifest(*positional, **keywords):
                report = produce(*positional, **keywords)
                (args.output_dir / "artifact-manifest.json").unlink()
                return report

            module.run.side_effect = without_manifest
            with self.assertRaises(FileNotFoundError):
                self.invoke(args, module)
            self.assert_empty_archive(args)

    def test_value_rejects_wrong_run_digest_and_unsafe_paths_before_archive(self):
        changes = {
            "run": lambda manifest: manifest.update(run_id="previous-run"),
            "digest": lambda manifest: manifest["files"].update({"provenance.json": "0" * 64}),
            "traversal": lambda manifest: manifest["files"].update({"../outside.json": "0" * 64}),
            "absolute": lambda manifest: manifest["files"].update({"/tmp/outside.json": "0" * 64}),
            "symlink": lambda manifest: manifest["files"].update({"escape.json": "0" * 64}),
        }
        for invalid, change in changes.items():
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args, module, _ = self.setup_run(root, "value-lab", manifest_change=change)
                outside = root / "outside.json"
                outside.write_bytes(b"{}")
                (args.output_dir / "escape.json").symlink_to(outside)
                with self.assertRaises(ValueError):
                    self.invoke(args, module)
                self.assert_empty_archive(args)
                self.assertEqual(outside.read_bytes(), b"{}")


class TestResearchCLI(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'optional calibration dependency')
    def test_offline_calibration_uses_checked_matching_snapshots_and_reports_failures(self):
        from datetime import date, timedelta
        from app.data_service.history import digest
        from app.research import main

        lines = ['Date,HomeTeam,AwayTeam,FTHG,FTAG']
        for i in range(90):
            day = (date(2024, 8, 1) + timedelta(days=i)).strftime('%d/%m/%Y')
            home, away = ((0, 1), (1, 1), (2, 0))[i % 3]
            lines.append(f'{day},A,B,{home},{away}')
        matches = FootballCSVClient().parse_csv_text('PL', '\n'.join(lines), 'https://public.invalid/snapshot.csv')
        snapshot = {'schema': 1, 'source': 'football-csv', 'provider': 'football-data.co.uk',
                    'competition': 'PL', 'season': '2024', 'fetched_at': '2026-10-17', 'matches': matches}
        snapshot['sha256'] = digest(snapshot)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'PL').mkdir()
            path = root / 'PL/2024.json'; path.write_text(json.dumps(snapshot))
            output = root / 'output/report.json'
            argv = ['calibrate', '--source', 'football-csv', '--competitions', 'PL', '--seasons', '2024',
                    '--model', 'logistic_l2', '--snapshots', str(root), '--output', str(output),
                    '--calibrators', 'identity,temperature,dirichlet_l2,draw_constant,draw_logistic,draw_beta']
            with patch('app.data_service.fetch.public.create_data_client', side_effect=AssertionError('offline')):
                self.assertEqual(main(argv), 0)
                report = json.loads(output.read_text())
                self.assertEqual(len(report['results']), 1)
                self.assertEqual(report['results'][0]['source_snapshots'][0]['sha256'], snapshot['sha256'])
                self.assertNotIn('private_predictions', report['results'][0])
                self.assertEqual(set(report['results'][0]['calibrators']),
                                 {'identity', 'temperature', 'dirichlet_l2', 'draw_constant', 'draw_logistic', 'draw_beta'})
                self.assertTrue(all(row['state'] == 'completed' for row in report['results'][0]['results']))
                for change, expected in [('corrupt', '数据快照校验失败'), ('identity', '与请求不符')]:
                    bad = dict(snapshot, competition='PD')
                    if change == 'identity':
                        bad['sha256'] = digest({k: v for k, v in bad.items() if k != 'sha256'})
                    path.write_text(json.dumps(bad))
                    self.assertEqual(main(argv), 1)
                    failed = json.loads(output.read_text())
                    self.assertFalse(failed['results'])
                    self.assertIn(expected, failed['errors'][0]['message'])

    def test_research_source_option_overrides_environment_for_analysis_and_experiments(self):
        from app.research import main
        from app.data_service.fetch.public import OpenFootballClient

        matches = FootballCSVClient().parse_csv_text(
            "PL",
            "Date,HomeTeam,AwayTeam,FTHG,FTAG\n01/08/2025,A,B,2,1\n",
            "https://public.invalid/example.csv",
        )
        cases = [
            ("openfootball", "football-csv", OpenFootballClient, FootballCSVClient),
            ("football-csv", "openfootball", FootballCSVClient, OpenFootballClient),
        ]
        for command in ("analyze", "experiment"):
            for selected, previous, chosen_client, other_client in cases:
                with self.subTest(command=command, source=selected):
                    output = StringIO()
                    with patch.dict(
                        os.environ,
                        {
                            "SOCCER_ANALYTICS_DATA_SOURCE": previous,
                            "FOOTBALL_DATA_API_KEY": "",
                        },
                    ), patch.object(
                        chosen_client, "fetch_matches", return_value=matches
                    ) as fetch, patch.object(
                        other_client,
                        "fetch_matches",
                        side_effect=AssertionError("wrong provider"),
                    ), patch(
                        "app.ml.experiments.run_experiments", return_value={"evaluated": True}
                    ), redirect_stdout(output):
                        result = main(
                            [
                                command,
                                "--source", selected,
                                "--competitions", "PL",
                                "--seasons", "2025",
                            ]
                        )
                    report = json.loads(output.getvalue())
                    self.assertEqual(result, 0, report["errors"])
                    self.assertEqual(report["source"], chosen_client.source_name)
                    fetch.assert_called_once_with(
                        "PL", season="2025", status="FINISHED"
                    )


    def test_match_study_isolates_leagues_and_keeps_private_errors_off_public_output(self):
        from app.cli.model_research import _run_match_study

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            args = SimpleNamespace(seasons=["2025"], competitions="PL,PD", output=output)
            settings = SimpleNamespace(training_seasons=["2024"])
            source = {"provider": "fixture", "sha256": "a" * 64}
            diagnostics, evaluated = StringIO(), []
            def evaluate(matches):
                evaluated.append(matches)
                return {"rows": len(matches)}
            with patch("app.cli.model_research.load_settings", return_value=settings), \
                 patch("app.cli.model_research.resolve_competitions", return_value={"PL": 1, "PD": 2}), \
                 patch("app.cli.model_research._match_research_source", return_value=(None, "football-csv")), \
                 patch("app.cli.model_research._match_research_matches", side_effect=[([{"id": 7}], [source]), ValueError("private-match-123")]), \
                 patch("sys.stderr", diagnostics):
                code = _run_match_study(args, evaluate, public_error="study failed; see local diagnostics")
            public = json.loads(output.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(evaluated, [[{"id": 7}]])
            self.assertEqual(public["results"], [{"competition": "PL", "source_snapshots": [source], "rows": 1}])
            self.assertEqual(public["errors"][0]["competition"], "PD")
            self.assertNotIn("private-match-123", output.read_text())
            self.assertIn("private-match-123", diagnostics.getvalue())
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])


class TestImportBoundaries(unittest.TestCase):
    def test_lightweight_entrypoints_do_not_load_unrelated_dependencies(self):
        cases = (
            ("research help", ("app.web", "app.ml.training", "xgboost", "sklearn", "torch"),
             "from app.research import main\nmain(['--help'])", ("representation-lab", "export-research")),
            ("public collectors", ("app.web", "xgboost", "sklearn"),
             "from app.data_service import archive, bodypose, history, bulk_history, recent_matches\nfrom app.analytics import movement, scouting", ()),
        )
        for label, blocked, entrypoint, expected in cases:
            with self.subTest(entrypoint=label):
                script = """import importlib.abc, sys
class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(BLOCKED):
            raise ImportError('Entry point loaded unrelated dependency: ' + fullname)
sys.meta_path.insert(0, BlockHeavyImports())
""".replace('BLOCKED', repr(blocked)) + entrypoint
                result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                for command in expected:
                    self.assertIn(command, result.stdout)

    def test_data_layer_never_imports_analysis_or_prediction(self):
        import ast

        violations = []
        for path in (APP_ROOT / "data_service").rglob("*.py"):
            for entry in ast.walk(ast.parse(path.read_text())):
                modules = [entry.module] if isinstance(entry, ast.ImportFrom) else [name.name for name in entry.names] if isinstance(entry, ast.Import) else []
                for module in modules:
                    if module and module.startswith(("app.ml", "app.analytics", "app.web", "app.cli")):
                        violations.append(f"{path.relative_to(ROOT)}:{entry.lineno}: {module}")
        self.assertEqual(violations, [], "Data ingestion must not load forecasting or analysis: " + ", ".join(violations))

    def test_verification_guard_blocks_files_and_sqlite_uris_before_access(self):
        script = r"""import pathlib, sqlite3, sys, tempfile
import scripts.check as check
with tempfile.TemporaryDirectory() as directory:
    check.ROOT = pathlib.Path(directory)
    (check.ROOT / 'data').mkdir()
    protected = check.ROOT / 'data/football.sqlite3'
    protected.write_bytes(b'fixture, not the project warehouse')
    allowed = check.ROOT / 'fixture.txt'
    allowed.write_text('allowed')
    sys.addaudithook(check.protect_warehouse)
    cases = [lambda: protected.read_bytes(),
             lambda: pathlib.Path(str(protected)+'-wal').open('wb'),
             lambda: sqlite3.connect(protected),
             lambda: sqlite3.connect(protected.as_uri()+'?mode=ro', uri=True)]
    for invoke in cases:
        try: invoke()
        except RuntimeError as error: assert 'production football warehouse' in str(error)
        else: raise AssertionError('Protected warehouse was reached')
    assert allowed.read_text() == 'allowed'
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_public_manifest_rejects_private_paths_and_symlink_ancestors(self):
        import scripts.public_manifest as manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "allowed").mkdir()
            (root / "allowed/file.json").write_text("{}")
            (root / "redirect").symlink_to(root / "allowed", target_is_directory=True)
            with patch.object(manifest, "ROOT", root):
                self.assertEqual(manifest.approved_file("allowed/file.json"), root / "allowed/file.json")
                for name in ["data/football.sqlite3", ".tmp/run/report.json", "models/model.joblib",
                             "allowed/data.npz", "../file.json", str(root / "allowed/file.json"),
                             "redirect/file.json"]:
                    with self.subTest(path=name), self.assertRaises(ValueError):
                        manifest.approved_file(name)
            (root / 'private.txt').write_text('must never be read')
            (root / 'results').mkdir()
            entry = root / 'results/public-manifest.json'
            entry.symlink_to(root / 'private.txt')
            for parent_link in (False, True):
                if parent_link:
                    entry.unlink()
                    entry.parent.rmdir()
                    entry.parent.symlink_to(root / 'allowed', target_is_directory=True)
                with self.subTest(manifest_parent_symlink=parent_link), \
                     patch.object(manifest, 'ROOT', root), patch.object(manifest, 'MANIFEST', entry), \
                     patch.object(Path, 'read_text', side_effect=AssertionError('Read before validation')), \
                     self.assertRaisesRegex(ValueError, 'Symlink'):
                    manifest.inspect()

    def test_all_application_modules_import(self):
        failures = []
        optional = {
            'app.analytics.representation.models': 'torch',
            'app.analytics.spatial_style': 'sklearn',
        }
        for path in sorted(APP_ROOT.rglob("*.py")):
            module = _module_name(path)
            with self.subTest(module=module):
                dependency = optional.get(module)
                if dependency and importlib.util.find_spec(dependency) is None:
                    self.skipTest(f'Optional module requires {dependency}')
                try:
                    importlib.import_module(module)
                except Exception as exc:
                    failures.append(f"{module}: {type(exc).__name__}: {exc}")
        self.assertEqual(failures, [], "\n".join(failures))


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)

def _python_version(contents: str, pattern: str) -> tuple[int, int]:
    match = re.search(pattern, contents, re.MULTILINE)
    if not match:
        raise AssertionError("运行配置缺少明确 Python major.minor")
    return int(match.group(1)), int(match.group(2))


class TestRuntimeConfiguration(unittest.TestCase):
    def test_execution_surfaces_share_python_and_the_canonical_pipeline(self):
        workflow = (ROOT / ".github/workflows/daily_update.yml").read_text()
        dockerfile = (ROOT / "Dockerfile").read_text()
        runner_path = ROOT / "run.sh"
        runner = runner_path.read_text(encoding="utf-8")

        self.assertEqual(
            _python_version(workflow, r'python-version:\s*"(\d+)\.(\d+)"'),
            _python_version(dockerfile, r"^FROM python:(\d+)\.(\d+)(?:[.-])"),
        )
        self.assertIn("FOOTBALL_DATA_API_KEY", workflow)
        self.assertIn("python -m app.pipeline all", workflow)
        self.assertTrue(runner_path.stat().st_mode & stat.S_IXUSR)
        self.assertIn("docker compose version", runner)
        self.assertLess(
            runner.index("docker compose version"),
            runner.index("docker-compose"),
        )

if __name__ == "__main__":
    unittest.main()
