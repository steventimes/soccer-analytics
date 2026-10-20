import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.analytics.artifacts import archive_outputs, file_digest
from app.data_service.database import export_asset
from app.data_service.warehouse import Warehouse


class TestArtifactSelection(unittest.TestCase):
    def test_archived_outputs_retain_explicit_input_relations(self):
        from app.data_service.database import analysis_lineage

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.json'
            source.write_text('{"source": true}')
            output = root / 'output'
            output.mkdir()
            (output / 'report.json').write_text('{"metric": 2}')
            with Warehouse(root / 'archive.sqlite3') as db:
                aid = db.asset(db.dataset('source', 'football', 'statsbomb'), source, 'source.json')
                archive_outputs(db, 'derived', output, 'research', 'one')
                db.record_analysis_inputs('research', 'one', (aid, aid))
                db.record_analysis_inputs('research', 'one', (aid,))
                with self.assertRaisesRegex(ValueError, 'different recorded inputs'):
                    other = db.asset(dataset_id=db.dataset('other', 'football', 'test'), path=source, relative_path='other.json')
                    db.record_analysis_inputs('research', 'one', (other,))
            lineage = analysis_lineage(root / 'archive.sqlite3', 'one')
            self.assertTrue(lineage['inputs_recorded'])
            self.assertEqual(lineage['inputs']['total'], 1)
            self.assertEqual(lineage['inputs']['assets'][0]['asset_id'], aid)
            self.assertEqual(lineage['outputs']['total'], 1)

    def test_invalid_input_relations_leave_archive_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'report.json').write_text('{}')
            with Warehouse(root / 'archive.sqlite3') as db:
                for ids in ((999,), (True,), ('1',), (0,)):
                    with self.subTest(ids=ids):
                        with self.assertRaises(ValueError):
                            db.record_analysis_inputs('research', 'one', ids)
                        self.assertEqual(db.connection.execute('SELECT COUNT(*) FROM datasets').fetchone()[0], 0)
                        self.assertEqual(db.connection.execute('SELECT COUNT(*) FROM assets').fetchone()[0], 0)

    def assert_jsonl_export(self, db, root, original, relative_name):
        row = db.connection.execute(
            'SELECT id,sha256 FROM assets WHERE path=?', (relative_name,)
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row['sha256'], file_digest(original))
        self.assertIsNone(db.connection.execute(
            'SELECT asset_id FROM documents WHERE asset_id=?', (row['id'],)
        ).fetchone())
        self.assertIsNone(db.connection.execute(
            'SELECT asset_id FROM json_lines WHERE asset_id=?', (row['id'],)
        ).fetchone())
        db.connection.commit()
        exported = root / 'exported-context.jsonl'
        export_asset(root / 'archive.sqlite3', row['id'], exported)
        self.assertEqual(exported.read_bytes(), original.read_bytes())
        self.assertEqual(file_digest(exported), row['sha256'])

    def test_explicit_selection_preserves_unselected_old_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output'
            output.mkdir()
            (output / 'current.json').write_text('{"run_id":"new"}')
            context = output / 'frame-context.jsonl'
            raw_context = b'{"frame":1,"players":[]}\r\n{"frame":2,"players":[3]}\n'
            context.write_bytes(raw_context)
            (output / 'seed-29.npz').write_bytes(b'old run artifact')
            with Warehouse(root / 'archive.sqlite3') as db:
                archive_outputs(db, 'new', output, 'value-lab', 'new', filenames=['current.json', 'frame-context.jsonl'])
                paths = {row[0] for row in db.connection.execute('SELECT path FROM assets')}
                self.assertEqual(paths, {'current.json', 'frame-context.jsonl'})
                self.assertEqual(db.connection.execute('SELECT run_id FROM analysis_run_assets').fetchone()[0], 'new')
                self.assert_jsonl_export(db, root, context, 'frame-context.jsonl')
            self.assertEqual(context.read_bytes(), raw_context)
            self.assertEqual((output / 'seed-29.npz').read_bytes(), b'old run artifact')

    def test_invalid_selection_is_rejected_before_any_archive_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output'
            output.mkdir()
            (output / 'current.json').write_text('{}')
            (output / 'unsupported.txt').write_text('not an artifact')
            outside = root / 'outside.json'
            outside.write_text('{}')
            (output / 'escape.json').symlink_to(outside)
            invalid = ['missing.json', '../outside.json', str(outside), 'escape.json', 'unsupported.txt']
            with Warehouse(root / 'archive.sqlite3') as db:
                for name in invalid:
                    with self.subTest(name=name):
                        with self.assertRaises(ValueError):
                            archive_outputs(db, 'new', output, 'value-lab', 'new', filenames=['current.json', name])
                        self.assertEqual(db.connection.execute('SELECT COUNT(*) FROM datasets').fetchone()[0], 0)
                        self.assertEqual(db.connection.execute('SELECT COUNT(*) FROM assets').fetchone()[0], 0)
                        self.assertIsNone(db.connection.execute("SELECT name FROM sqlite_master WHERE name='analysis_run_assets'").fetchone())

    def test_default_keeps_existing_recursive_archive_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output'
            (output / 'nested').mkdir(parents=True)
            (output / 'current.json').write_text('{}')
            (output / 'nested' / 'model.npz').write_bytes(b'numeric archive placeholder')
            context = output / 'nested' / 'frame-context.jsonl'
            raw_context = b'{"frame":10,"players":[]}\n{"frame":11,"players":[5]}\n'
            context.write_bytes(raw_context)
            (output / 'ignored.txt').write_text('not included')
            with Warehouse(root / 'archive.sqlite3') as db:
                archive_outputs(db, 'legacy', output, 'legacy', 'run')
                paths = {row[0] for row in db.connection.execute('SELECT path FROM assets')}
                self.assertEqual(paths, {'current.json', 'nested/model.npz', 'nested/frame-context.jsonl'})
                self.assert_jsonl_export(db, root, context, 'nested/frame-context.jsonl')
            self.assertEqual(context.read_bytes(), raw_context)

    def test_rerun_seed_and_cap_change_only_manifests_and_archives_current_outputs(self):
        try:
            import torch
            import sklearn
        except ImportError:
            self.skipTest('Optional experiment dependencies not installed')
        from app.analytics.action_value import run

        def tiny_data(database, league, cap, horizon):
            data = {}
            for offset, split in enumerate(('train', 'validation', 'test')):
                labels = np.column_stack((np.arange(cap) % 2, (np.arange(cap) // 2) % 2)).astype(np.float32)
                data[split] = (np.full((cap, 8), 8, dtype=np.int64),
                               np.full((cap, 8, 4), .4, dtype=np.float32), labels,
                               np.arange(cap, dtype=np.int64) + offset * 100,
                               np.full(cap, offset + 1, dtype=np.int64))
            return data, {'provider': 'wyscout', 'test_fixture': True}, {'grid': np.zeros((8, 12)).tolist()}

        with tempfile.TemporaryDirectory() as directory, patch('app.analytics.action_value.load_data', side_effect=tiny_data):
            root = Path(directory)
            output = root / 'output'
            with redirect_stdout(StringIO()):
                old = run('unused', output, train_cap=12, epochs=1, seeds=(11, 29))
                old_bytes = (output / 'gru_context-seed-29.npz').read_bytes()
                current = run('unused', output, train_cap=8, epochs=1, seeds=(11,))
            self.assertNotEqual(old['run_id'], current['run_id'])
            manifest = json.loads((output / 'artifact-manifest.json').read_text())
            self.assertEqual(manifest['run_id'], current['run_id'])
            self.assertTrue(manifest['files'])
            self.assertFalse(any('seed-29' in name for name in manifest['files']))
            self.assertEqual((output / 'gru_context-seed-29.npz').read_bytes(), old_bytes)
            with np.load(output / 'train-examples.npz', allow_pickle=False) as arrays:
                self.assertEqual(len(arrays['event_id']), 8)
            with Warehouse(root / 'archive.sqlite3') as db:
                selected = [*manifest['files'], 'artifact-manifest.json']
                archive_outputs(db, 'current', output, 'value-lab', current['run_id'], filenames=selected)
                archived = {row[0] for row in db.connection.execute('SELECT a.path FROM assets a JOIN analysis_run_assets r ON r.asset_id=a.id WHERE r.run_id=?', (current['run_id'],))}
                self.assertEqual(archived, set(selected))
                self.assertFalse(any('seed-29' in name for name in archived))


if __name__ == '__main__':
    unittest.main()
