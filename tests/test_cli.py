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
    def test_forecast_registration_keeps_identity_and_rejects_late_invalid_or_overwritten_batches(self):
        import copy
        import hashlib
        from datetime import UTC, datetime
        from app.ml.forecast_identity import forecast_id_for_prediction, forecast_id_matches
        from app.ml.study_protocol import register_forecasts
        from app.research import main
        from tests.support import _valid_prediction

        prediction = _valid_prediction(1)
        report = {'generated_at':'2026-07-09T00:00:00Z','predictions':[prediction],'errors':[]}
        now = datetime(2026, 7, 9, 1, tzinfo=UTC)
        sealed = register_forecasts(report, input_cutoff='2026-07-08T23:00:00Z', input_sha256='a'*64, now=now)
        self.assertTrue(forecast_id_matches(sealed['predictions'][0]))
        self.assertFalse(sealed['prospective_accuracy_verified'])
        self.assertEqual(sealed['recorded_at'], now.isoformat())
        for change in ('late', 'duplicate', 'tampered', 'unnormalized', 'future-input'):
            bad = copy.deepcopy(report)
            if change == 'late': bad['predictions'][0]['utc_date']='2026-07-09T01:00:00Z'
            if change == 'duplicate': bad['predictions'].append(copy.deepcopy(prediction))
            if change == 'tampered': bad['predictions'][0]['confidence']=0.9
            if change == 'unnormalized':
                bad['predictions'][0]['probabilities']={'home_win':0.7,'draw':0.6,'away_win':0.3}
            if change in {'late', 'unnormalized'}:
                bad['predictions'][0]['forecast_id']=forecast_id_for_prediction(bad['predictions'][0])
            if change == 'future-input': bad['generated_at']='2026-07-10T00:00:00Z'
            with self.subTest(change=change), self.assertRaises(ValueError):
                register_forecasts(bad, input_cutoff='2026-07-08T23:00:00Z', input_sha256='a'*64, now=now)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source=root/'forecasts.json'; target=root/'registration.json'
            source.write_text(json.dumps(report))
            argv=['record-forecasts','--predictions',str(source),'--input-cutoff','2026-07-08T23:00:00Z','--output',str(target)]
            old_mask=os.umask(0o022)
            try:
                with patch('app.ml.study_protocol.datetime') as clock, redirect_stdout(StringIO()) as stdout:
                    clock.now.return_value=now
                    self.assertEqual(main(argv), 0)
                    original=target.read_bytes()
                    self.assertEqual(stat.S_IMODE(target.stat().st_mode),0o600,'forecast registration must be private under umask 022')
                    with self.assertRaises(FileExistsError): main(argv)
                    self.assertEqual(target.read_bytes(),original,'duplicate registration cannot replace the original')
            finally:
                os.umask(old_mask)
            saved=json.loads(original)
            self.assertEqual(saved['recorded_at'],now.isoformat())
            self.assertEqual(saved['generated_at'],'2026-07-09T00:00:00+00:00')
            self.assertEqual(saved['input_cutoff'],'2026-07-08T23:00:00+00:00')
            self.assertEqual(saved['predictions'][0]['forecast_id'],prediction['forecast_id'])
            self.assertFalse(saved['prospective_accuracy_verified'])
            self.assertEqual(saved['input_sha256'],hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(saved['run_receipt']['inputs']['forecast_file_sha256'],saved['input_sha256'])
            self.assertTrue(forecast_id_matches(saved['predictions'][0]))
            self.assertEqual(json.loads(stdout.getvalue())['sha256'],saved['sha256'])
            with self.assertRaisesRegex(ValueError, '私有目录'):
                main([*argv[:-1], str(ROOT/'docs/forbidden-registration.json')])

    MODULES = {
        "passing-lab": 'app.analytics.tactics.passing_context',
        "space-lab": 'app.analytics.tactics.receiving_space',
        "value-lab": 'app.analytics.events.action_value',
        "buildup-lab": 'app.analytics.tactics.buildup_roles',
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
    def test_data_compaction_roundtrip_resume_and_failed_publication_preserve_original_bytes(self):
        import gzip
        import hashlib
        from scripts.compact_data import compact_file
        from app.data_service.archive import open_asset_file

        payload=b'{"events":['+b'"original content",'*4000+b'"last"]}\n'
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'source.json'; source.write_bytes(payload); source.chmod(0o640)
            entry={'path':str(source),'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'kind':'source'}
            with patch('scripts.compact_data._sync_directory',side_effect=OSError('directory sync failed')):
                with self.assertRaises(OSError): compact_file(entry)
            self.assertEqual(source.read_bytes(),payload)
            target=Path(str(source)+'.gz')
            self.assertEqual(gzip.decompress(target.read_bytes()),payload)
            compressed=target.read_bytes()
            result=compact_file(entry)
            self.assertEqual(result['state'],'duplicate_plain_retired')
            self.assertFalse(source.exists())
            self.assertEqual(target.read_bytes(),compressed)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode),0o640)
            with open_asset_file(source) as handle: self.assertEqual(handle.read(),payload)
            self.assertEqual(compact_file(entry)['state'],'already_compressed')
            fresh=Path(temporary)/'fresh.json'; fresh.write_bytes(payload)
            self.assertEqual(compact_file(dict(entry,path=str(fresh)))['state'],'compressed')
            self.assertFalse(fresh.exists())
            self.assertEqual(gzip.decompress(Path(str(fresh)+'.gz').read_bytes()),payload)
            self.assertFalse(list(Path(temporary).glob('*.part')))

    def test_data_compaction_rejects_changed_or_conflicting_sources_without_destroying_either_copy(self):
        import gzip
        import hashlib
        from scripts.compact_data import compact_file

        payload=b'original bytes\n'*5000
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'data.json'; source.write_bytes(payload)
            entry={'path':str(source),'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'kind':'cache'}
            with self.assertRaises(ValueError): compact_file(dict(entry,sha256='a'*64))
            self.assertEqual(source.read_bytes(),payload)
            self.assertFalse(Path(str(source)+'.gz').exists())
            real_open=gzip.open
            changed=payload+b'new observation'
            def during_validation(*args,**kwargs):
                source.write_bytes(changed)
                return real_open(*args,**kwargs)
            with patch('scripts.compact_data.gzip.open',side_effect=during_validation):
                with self.assertRaisesRegex(ValueError,'changed'): compact_file(entry)
            self.assertEqual(source.read_bytes(),changed)
            self.assertFalse(Path(str(source)+'.gz').exists())
            source.write_bytes(payload)
            target=Path(str(source)+'.gz'); target.write_bytes(gzip.compress(b'other unique data'))
            with self.assertRaises(ValueError): compact_file(entry)
            self.assertEqual(source.read_bytes(),payload)
            self.assertEqual(gzip.decompress(target.read_bytes()),b'other unique data')
            self.assertFalse(list(Path(temporary).glob('.*.part')))

    def test_compaction_plan_cannot_overwrite_aliases_or_size_filtered_inputs(self):
        import hashlib
        import gzip
        from scripts.compact_data import run, candidates

        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/'data.json'; raw=b'{"small":"original"}'
            source.write_bytes(raw)
            receipt=source.with_suffix('.receipt.json'); body=json.dumps({'content_sha256':hashlib.sha256(raw).hexdigest()})
            receipt.write_text(body)
            for minimum in (0,1000000):
                for target in (source,receipt):
                    alias=root/'absent'/'..'/target.name
                    args=SimpleNamespace(source_root=[],cache=[source],minimum_bytes=minimum,
                        level=3,workers=1,report=alias,apply=False)
                    with self.subTest(minimum=minimum,target=target), self.assertRaisesRegex(ValueError,'overwrite'):
                        run(args)
                    self.assertEqual(source.read_bytes(),raw)
                    self.assertEqual(receipt.read_text(),body)
            output=root/'inventory.json'
            args.report=output
            self.assertEqual(run(args)['state'],'planned')
            self.assertTrue(source.exists())
            native=Path(str(source)+'.gz'); native.write_bytes(gzip.compress(raw))
            manifest={'complete':True,'dataset':'statsbomb','files':[
                {'path':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'bytes':file.stat().st_size}
                for file in (source,native)]}
            (root/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(candidates([root],[],0)[0],[])
            self.assertEqual(source.read_bytes(),raw)

    def test_search_admission_is_offline_and_blocked_execution_cannot_publish_or_overwrite_inputs(self):
        from app.data_service.history import digest
        from app.research import main
        from tests.test_core import _random_matches

        rows = _random_matches(11, 60)
        for row in rows:
            row['result_available_at'] = '2026-10-23T12:00:00Z'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); snapshots=root/'snapshots'; (snapshots/'PL').mkdir(parents=True)
            snapshot={'schema':1,'source':'football-data','provider':'football-data.org','competition':'PL',
                'season':'2025','fetched_at':'2026-10-23T12:00:00Z','matches':rows}
            snapshot['sha256']=digest(snapshot)
            path=snapshots/'PL/2025.json'; path.write_text(json.dumps(snapshot))
            output=root/'admission.json'
            argv=['search-comparison','--source','football-data','--snapshots',str(snapshots),
                '--competitions','PL','--seasons','2025','--as-of','2026-10-25T00:00:00Z',
                '--evaluation-start','2026-10-17T00:00:00Z','--evaluation-end','2026-10-22T00:00:00Z',
                '--final-holdout-start','2026-10-24T00:00:00Z','--output',str(output)]
            before=path.read_bytes()
            with patch('app.data_service.fetch.public.create_data_client',side_effect=AssertionError('offline')):
                self.assertEqual(main([*argv,'--admit-only']),0)
                report=json.loads(output.read_text())
                self.assertEqual(report['results'][0]['status'],'blocked')
                self.assertEqual(report['results'][0]['admission']['readiness']['state'],'unavailable')
                self.assertEqual(report['results'][0]['runs'],[])
                self.assertEqual(main(argv),1)
                self.assertEqual(json.loads(output.read_text())['errors'][0]['code'],'SearchIncomplete')
                private=root/'private-predictions.json'
                old_mask=os.umask(0o022)
                try:
                    self.assertEqual(main([*argv,'--private-predictions',str(private)]),1)
                    self.assertEqual(stat.S_IMODE(private.stat().st_mode),0o600,'search audit must be private under umask 022')
                    self.assertEqual(stat.S_IMODE(output.stat().st_mode),0o644,'public aggregate permissions must stay unchanged')
                    self.assertEqual(json.loads(private.read_text()),{'schema':'SoccerAnalytics.PrivateSearchPredictions.v1','rows':[]})
                    audit_bytes=private.read_bytes();report_bytes=output.read_bytes()
                    with self.assertRaises(FileExistsError): main([*argv,'--private-predictions',str(private)])
                    self.assertEqual(private.read_bytes(),audit_bytes)
                    self.assertEqual(output.read_bytes(),report_bytes)
                finally:
                    os.umask(old_mask)
                with self.assertRaisesRegex(ValueError,'不能覆盖'):
                    main([*argv,'--private-predictions',str(path)])
            self.assertEqual(path.read_bytes(),before)

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'optional research dependency')
    def test_offline_ale_and_season_study_dispatch_with_timing_and_registration(self):
        from datetime import UTC, datetime, timedelta
        from app.data_service.history import digest
        from app.research import main
        from tests.test_core import _random_matches
        rows = _random_matches(11, 90)
        for row in rows:
            row['result_available_at'] = (datetime.fromisoformat(row['utcDate']) + timedelta(hours=2)).isoformat()
            row['homeTeam']['name'] = str(row['homeTeam']['id'])
            row['awayTeam']['name'] = str(row['awayTeam']['id'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'PL').mkdir()
            snapshot = {'schema': 1, 'source': 'football-data', 'provider': 'football-data.org',
                'competition': 'PL', 'season': '2025', 'fetched_at': '2026-10-21T00:00:00Z', 'matches': rows}
            snapshot['sha256'] = digest(snapshot)
            (root / 'PL/2025.json').write_text(json.dumps(snapshot))
            argv = ['ale', '--source', 'football-data', '--competitions', 'PL', '--seasons', '2025',
                '--snapshots', str(root), '--model', 'logistic_l2', '--as-of', '2026-01-01T00:00:00Z',
                '--output', str(root / 'ale.json')]
            with patch('app.data_service.fetch.public.create_data_client', side_effect=AssertionError('offline')):
                self.assertEqual(main(argv), 0)
            report = json.loads((root / 'ale.json').read_text())
            self.assertEqual(report['results'][0]['timing_policy'], 'observed_result')
            self.assertEqual(report['results'][0]['training_rows'], 72)
            self.assertEqual(report['results'][0]['rows'], 18)
            manifest = {'collections': [{'path': 'PL/2025.json', 'sha256': snapshot['sha256'],
                'source': 'football-data', 'competition': 'PL', 'season': '2025'}]}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            protocol_path = root / 'future.json'
            start = (datetime.now(UTC) + timedelta(days=365)).isoformat()
            end = (datetime.now(UTC) + timedelta(days=730)).isoformat()
            freeze = ['season-study', '--history', str(root), '--test-seasons', '2027',
                '--as-of', '2026-01-01T00:00:00Z', '--freeze-protocol', str(protocol_path),
                '--evaluation-start', start, '--evaluation-end', end]
            with redirect_stdout(StringIO()): self.assertEqual(main(freeze), 0)
            before = protocol_path.read_bytes()
            with self.assertRaises(FileExistsError): main(freeze)
            self.assertEqual(protocol_path.read_bytes(), before)
            with self.assertRaises(ValueError):
                main(['season-study', '--history', str(root), '--test-seasons', '2027',
                      '--as-of', '2029-01-01T00:00:00Z', '--protocol', str(protocol_path),
                      '--output', str(protocol_path)])
            self.assertEqual(protocol_path.read_bytes(), before)
            import copy
            future_rows = copy.deepcopy(rows)
            for row in future_rows:
                row['id'] += 1000
                kickoff = datetime.fromisoformat(row['utcDate']).replace(year=2026)
                row['utcDate'] = kickoff.isoformat()
                row['result_available_at'] = (kickoff + timedelta(hours=2)).isoformat()
            future_snapshot = {**snapshot, 'season': '2026', 'matches': future_rows}
            future_snapshot['sha256'] = digest({k: v for k, v in future_snapshot.items() if k != 'sha256'})
            (root / 'PL/2026.json').write_text(json.dumps(future_snapshot))
            manifest['collections'].append({'path': 'PL/2026.json', 'sha256': future_snapshot['sha256'],
                'source': 'football-data', 'competition': 'PL', 'season': '2026'})
            (root / 'manifest.json').write_text(json.dumps(manifest))
            evaluate = ['season-study', '--history', str(root), '--test-seasons', '2026',
                '--as-of', '2027-01-01T00:00:00Z', '--output', str(root / 'season.json')]
            with patch('app.data_service.fetch.public.create_data_client', side_effect=AssertionError('offline')):
                self.assertEqual(main(evaluate), 0)
            season = json.loads((root / 'season.json').read_text())
            self.assertEqual(season['results'][0]['test_rows'], 90)
            self.assertEqual(season['protocol']['result_availability'], 'observed_result_strict_before_fit_and_kickoff')

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
            from app.run_receipts import content_digest
            self.assertEqual(public["run_receipt"]["inputs"]["cohorts"][0]["input_sha256"],
                             content_digest([{"id": 7}]))
            self.assertEqual(public["run_receipt"]["stages"], {"PL": "complete", "PD": "failed"})
            self.assertEqual(public["errors"][0]["competition"], "PD")
            self.assertNotIn("private-match-123", output.read_text())
            self.assertIn("private-match-123", diagnostics.getvalue())
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])


class TestImportBoundaries(unittest.TestCase):
    def test_missing_optional_analysis_packages_keep_base_numeric_checks_active(self):
        script = "import importlib.abc, importlib.util, sys, unittest\nfrom scripts.check import protect_warehouse\nsys.addaudithook(protect_warehouse)\nblocked = {'sklearn', 'matplotlib', 'threadpoolctl'}\nclass MissingOptionalModules(importlib.abc.MetaPathFinder):\n    def find_spec(self, fullname, path=None, target=None):\n        if fullname.split('.')[0] in blocked:\n            raise ModuleNotFoundError('Simulated missing optional package: ' + fullname, name=fullname)\noriginal_find_spec = importlib.util.find_spec\ndef find_spec(name, *args, **kwargs):\n    return None if name.split('.')[0] in blocked else original_find_spec(name, *args, **kwargs)\nsys.meta_path.insert(0, MissingOptionalModules())\nimportlib.util.find_spec = find_spec\nnames = [\n    'tests.analytics.test_scouting.TestFootballFigures.test_real_network_figure_renders_both_pass_directions',\n    'tests.ml.test_representation.TestRepresentationModels.test_spatial_nmf_random_starts_are_reproducible_and_fit_early_only',\n    'tests.analytics.test_scouting.TestAssessments.test_spatial_nmf_basis_never_fits_later_period',\n    'tests.analytics.test_action_value.TestActionValue.test_calibration_ties_are_pooled_and_brier_decomposition_is_exact',\n    'tests.ml.test_representation.TestRepresentationModels.test_spatial_basis_matching_handles_permutations_and_dead_components',\n    'tests.analytics.test_scouting.TestAssessments.test_spatial_retrieval_shares_ties_and_respects_candidate_pools',\n    'tests.analytics.test_action_value.TestActionValue.test_calibration_rejects_invalid_or_empty_probabilities',\n    'tests.ml.test_experiments.TestProbabilityDiagnostics.test_brier_partition_identity_signed_remainders_and_display_independence',\n]\nsuite = unittest.defaultTestLoader.loadTestsFromNames(names)\nresult = unittest.TextTestRunner(verbosity=2).run(suite)\nassert result.testsRun == 8, result.testsRun\nassert len(result.skipped) == 4, result.skipped\nassert result.wasSuccessful(), result.errors\n"
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_lightweight_entrypoints_do_not_load_unrelated_dependencies(self):
        cases = (
            ("research help", ("app.web", "app.ml.training", "xgboost", "sklearn", "torch"),
             "from app.research import main\nmain(['--help'])", ("representation-lab", "export-research")),
            ("public collectors", ("app.web", "xgboost", "sklearn"),
             "from app.data_service import archive, bodypose, history, bulk_history, recent_matches\nfrom app.analytics.tracking import movement\nfrom app.analytics.scouting import metrics as scouting", ()),
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
                             "AGENTS.md", ".codex/guidance/readme.md", ".agents/local.md",
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


class TestIdentityBridgeCLI(unittest.TestCase):
    def test_offline_audit_cli_preserves_inputs_and_rejects_output_collisions(self):
        from app.research import main
        from tests.data.test_ingestion import TestIdentityBridgeAudit
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);root=base/'release';root.mkdir()
            queries,sha=TestIdentityBridgeAudit()._identity_fixture(root)
            output=base/'audit.json'
            argv=['audit-identity-bridges','--release-dir',str(root),'--release-sha256',sha,
                  '--queries',str(queries),'--output',str(output)]
            with patch('requests.Session',side_effect=AssertionError('offline CLI')), redirect_stdout(StringIO()):
                self.assertEqual(main(argv),0)
            report=json.loads(output.read_text())
            self.assertEqual(report['counts']['unique'],2)
            self.assertEqual(report['adoption'],'not_adopted')
            with self.assertRaises(FileExistsError):main(argv)
            for path in (queries,root/'release.json',ROOT/'docs/private-identity-test.json'):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError,'private'):
                    main(argv[:-1]+[str(path)])
            untouched=base/'failure.json'
            with self.assertRaisesRegex(ValueError,'Release checksum'):
                main([v if v!=sha else '0'*64 for v in argv[:-1]]+[str(untouched)])
            self.assertFalse(untouched.exists())


class TestObservedHistoryCLI(unittest.TestCase):
    def test_prepare_observations_is_private_offline_and_readable_by_research(self):
        import hashlib
        from app.research import main
        from app.data_service.history import read_snapshot
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'input.csv'
            raw=b'Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG\nE0,01/08/2026,15:00,A,B,2,1\n'
            path.write_bytes(raw);path.with_suffix('.csv.receipt.json').write_text(json.dumps({
                'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'observed_at':'2026-08-02T00:00:00Z',
                'url':'https://www.football-data.co.uk/mmz4281/2627/E0.csv'}))
            output=root/'derived';argv=['prepare-observed-history','--csv',str(path),'--competition','PL',
                '--season','2026','--as-of','2026-08-04T00:00:00Z','--selection-as-of','2026-08-03T00:00:00Z',
                '--output-dir',str(output)]
            with patch('requests.sessions.Session.request',side_effect=AssertionError('no network')),redirect_stdout(StringIO()):
                self.assertEqual(main(argv),0)
            result=read_snapshot(output/'football-csv/PL/2026.json')
            self.assertEqual(len(result['matches']),1)
            self.assertEqual(result['matches'][0]['result_observed_at'],'2026-08-02T00:00:00+00:00')
            with self.assertRaises(ValueError):main(argv)
            with self.assertRaisesRegex(ValueError,'private'):main(argv[:-1]+[str(ROOT/'docs/forbidden-cohort')])
