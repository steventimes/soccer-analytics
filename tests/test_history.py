import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from app import pipeline
from app.data_service.fetch.public import FootballCSVClient


def match(
    index,
    year,
    *,
    home="Arsenal",
    away="Chelsea",
    score=(2, 1),
    source="football-data.co.uk",
):
    return {
        "id": year * 1000 + index,
        "utcDate": f"{year}-08-{1+index%28:02}T14:00:00+00:00",
        "source_date": f"{year}-08-{1+index%28:02}",
        "status": "FINISHED",
        "source": source,
        "source_url": "https://example.org/public-season",
        "homeTeam": {"id": 1, "name": home},
        "awayTeam": {"id": 2, "name": away},
        "score": {"fullTime": {"home": score[0], "away": score[1]}},
    }


class TestHistory(unittest.TestCase):
    def test_optional_xg_survives_normalization_without_entering_pre_match_features(self):
        import copy
        from app.ml.feature_engineering import FeatureEngineer

        header = "Date,HomeTeam,AwayTeam,FTHG,FTAG,HxG,AxG\n"
        csv = header + (
            "01/08/2026,Arsenal,Chelsea,2,1,0,2.75\n"
            "02/08/2026,Chelsea,Arsenal,1,1, ,1.2\n"
            "03/08/2026,Arsenal,Chelsea,0,0,-0.1,nan\n"
            "04/08/2026,Chelsea,Arsenal,1,2,1e309,typo\n"
            "05/08/2026,Arsenal,Chelsea,,,1.1,0.8\n"
        )
        rows = FootballCSVClient().parse_csv_text("PL", csv, "https://example.org/season.csv")
        self.assertEqual(len(rows), 5)  # Bad optional statistics never remove a fixture.
        self.assertEqual([(r['statistics']['home_xg'], r['statistics']['away_xg']) for r in rows],
                         [(0.0, 2.75), (None, 1.2), (None, None), (None, None), (1.1, 0.8)])
        self.assertEqual([(r['statistics_metadata']['xg']['home_status'],
                           r['statistics_metadata']['xg']['away_status']) for r in rows],
                         [('valid', 'valid'), ('missing', 'valid'), ('invalid', 'invalid'),
                          ('invalid', 'invalid'), ('valid', 'valid')])
        self.assertEqual(rows[-1]['status'], 'SCHEDULED')
        one_column = FootballCSVClient().parse_csv_text(
            'PL', 'Date,HomeTeam,AwayTeam,FTHG,FTAG,HxG\n01/08/2026,Arsenal,Chelsea,2,1,0\n', 'x')[0]
        self.assertEqual(one_column['statistics_metadata']['xg']['away_status'], 'column_absent')
        self.assertFalse(one_column['statistics_metadata']['xg']['complete_pair'])

        # Reparse the same fixtures in the historical schema: old payloads gain no keys.
        legacy_csv = '\n'.join(','.join(line.split(',')[:5]) for line in csv.splitlines()) + '\n'
        legacy = FootballCSVClient().parse_csv_text('PL', legacy_csv, 'https://example.org/season.csv')
        stripped = copy.deepcopy(rows)
        for row in stripped:
            row.pop('statistics_metadata')
            row['statistics'].pop('home_xg')
            row['statistics'].pop('away_xg')
        self.assertEqual(stripped, legacy)
        baseline = FeatureEngineer().build_training_dataset(legacy)
        self.assertEqual(len(baseline.rows[0]), 30)
        self.assertEqual(FeatureEngineer().build_training_dataset(rows), baseline)
        changed = copy.deepcopy(rows)
        for row in changed:
            row['statistics'].update(home_xg=999.0, away_xg=0.0)
            row['statistics_metadata']['xg']['complete_pair'] = False
        self.assertEqual(FeatureEngineer().build_training_dataset(changed), baseline)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "Install optional ML experiment dependencies")
    def test_regularized_models_complete_collinear_local_and_pooled_fits(self):
        from app.ml.seasonal import run_season_matrix

        collections = {('football-csv', code, str(year)):
                       [match(i, year, score=(i % 3, 1)) for i in range(40)]
                       for code in ('PL', 'SA') for year in (2020, 2021)}
        report = run_season_matrix(
            collections, test_seasons=['2021'], windows=['1'],
            models=['logistic_l1', 'logistic_l2', 'logistic_elasticnet'],
        )
        self.assertEqual(report['errors'], [])
        self.assertEqual(len(report['results']), 12)

    def test_local_and_pooled_xgboost_fits_receive_times(self):
        from app.ml.model import XGBoostClassifier
        from app.ml.seasonal import run_season_matrix

        fits = []
        class CapturingClassifier(XGBoostClassifier):
            def __init__(self, names):
                super().__init__(names, rounds=2)

            def fit(self, rows, targets, *, kickoffs=()):
                fits.append((len(rows), tuple(kickoffs)))
                return super().fit(rows, targets, **({'kickoffs': kickoffs} if kickoffs else {}))

        collections = {('football-csv', code, str(year)):
                       [match(i, year, score=(i % 3, 1)) for i in range(40)]
                       for code in ('PL', 'SA') for year in (2020, 2021)}
        report = run_season_matrix(collections, test_seasons=['2021'], windows=['1'],
                                  models=['xgboost'],
                                  factory=lambda *args, **kwargs: CapturingClassifier)
        self.assertEqual(report['errors'], [])
        self.assertEqual([size for size, _ in fits], [80, 40, 40])
        for size, times in fits:
            self.assertEqual(len(times), size)
            self.assertEqual(list(times), sorted(times))
            self.assertTrue(all(t.startswith('2020-') for t in times))

    def test_snapshot_is_reused_and_corruption_fails_closed(self):
        from app.data_service.history import collect_history, read_snapshot

        client = Mock(source_name="football-data.co.uk")
        client.fetch_matches.return_value = [match(1, 2020)]
        with tempfile.TemporaryDirectory() as directory:
            first = collect_history(
                Path(directory),
                ["football-csv"],
                ["PL"],
                ["2020"],
                client_factory=lambda _: client,
            )
            client.fetch_matches.side_effect = AssertionError(
                "cached collection must not download"
            )
            second = collect_history(
                Path(directory),
                ["football-csv"],
                ["PL"],
                ["2020"],
                client_factory=lambda _: client,
            )
            self.assertEqual(
                first["collections"][0]["sha256"], second["collections"][0]["sha256"]
            )
            path = Path(directory) / "football-csv" / "PL" / "2020.json"
            payload = json.loads(path.read_text())
            payload["matches"][0]["score"]["fullTime"]["home"] = 9
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                read_snapshot(path)

    def test_snapshot_ttl_expiry_rollover_bad_timestamps_and_failure_preserve_old_bytes(self):
        from datetime import UTC, datetime, timedelta, timezone
        from app.data_service.freshness import SnapshotCachePolicy
        from app.data_service.history import collect_history

        now = datetime(2026,10,17,12,tzinfo=UTC)
        policy = SnapshotCachePolicy(24, 6)
        self.assertEqual(policy.decide((now-timedelta(hours=5)).isoformat(),'2026',now=now)['state'],'fresh')
        self.assertEqual(policy.decide((now-timedelta(hours=6)).isoformat(),'2026',now=now)['state'],'expired')
        self.assertEqual(policy.decide((now-timedelta(hours=7)).astimezone(timezone(timedelta(hours=2))).isoformat(),'2025',now=now)['state'],'fresh')
        for stamp in [None, 'invalid', '2026-10-17T11:00:00', (now+timedelta(seconds=1)).isoformat()]:
            self.assertEqual(policy.decide(stamp,'2026',now=now)['state'],'expired')
            self.assertEqual(SnapshotCachePolicy().decide(stamp,'2026',now=now)['state'],'pinned')
        for day,state in [(datetime(2026,6,30,12,tzinfo=UTC),'expired'),(datetime(2026,7,1,12,tzinfo=UTC),'fresh')]:
            self.assertEqual(policy.decide((day-timedelta(hours=7)).isoformat(),'2025',now=day)['state'],state)
        self.assertEqual(SnapshotCachePolicy(0).decide(now.isoformat(),'2026',now=now)['state'],'expired')
        for value in [-1,True,float('nan'),float('inf'),1e308,10**1000]:
            with self.assertRaises(ValueError): SnapshotCachePolicy(value)
        client=Mock(source_name='football-data.co.uk')
        client.fetch_matches.side_effect=lambda code,season,status:[match(1,int(season))]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            def run(hours=0, **options):
                return collect_history(root,['football-csv'],['PL'],['2024','2026'],client_factory=lambda _:client,
                    active_max_age_hours=6,clock=lambda:now+timedelta(hours=hours),**options)
            first=run();self.assertFalse(first['errors']);self.assertEqual(client.fetch_matches.call_count,2)
            old=(root/'football-csv/PL/2024.json').read_bytes();active=(root/'football-csv/PL/2026.json').read_bytes()
            run(5);self.assertEqual(client.fetch_matches.call_count,2)
            expired=run(6);self.assertEqual(client.fetch_matches.call_count,3)
            self.assertEqual([c['cache']['state'] for c in expired['collections']],['pinned','expired'])
            self.assertEqual((root/'football-csv/PL/2024.json').read_bytes(),old)
            self.assertNotEqual((root/'football-csv/PL/2026.json').read_bytes(),active)
            saved={p:p.read_bytes() for p in root.glob('football-csv/PL/*.json')}
            client.fetch_matches.side_effect=TimeoutError('source unavailable')
            failed=run(12,max_age_hours=0)
            self.assertEqual(failed['collections'],[]);self.assertEqual(len(failed['errors']),2)
            self.assertTrue(all(e['cache']['state']=='expired' for e in failed['errors']))
            for p,before in saved.items():self.assertEqual(p.read_bytes(),before)

    def test_zip_ttl_refreshes_both_layers_repairs_legacy_and_rejects_corrupt_replacement(self):
        from datetime import UTC, datetime, timedelta
        import io,zipfile
        from app.data_service.bulk_history import collect_all_csv, INDEX_URL
        from app.data_service.history import digest, load_history

        now=datetime(2026,10,17,12,tzinfo=UTC)
        url='https://www.football-data.co.uk/mmz4281/2627/all-euro-data.zip'
        def zipped(score):
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w') as z:
                for code in ['E0','F1']:
                    z.writestr(code+'.csv',f'Date,HomeTeam,AwayTeam,FTHG,FTAG\n01/08/2026,A,B,{score},1\n')
            return buffer.getvalue()
        payload=zipped(2);calls=[];responses=[]
        def request(session,target):
            calls.append(target)
            response=Mock(text=f'<a href="{url}">CSV ZIP</a>',content=payload)
            responses.append(response);return response
        with tempfile.TemporaryDirectory() as temporary,patch('app.data_service.bulk_history._request',side_effect=request):
            root=Path(temporary)
            def run(hours=0, **options):
                return collect_all_csv(root,clock=lambda:now+timedelta(hours=hours),**options)
            first=run(active_max_age_hours=6);self.assertTrue(first['complete']);self.assertEqual(calls.count(url),1)
            snapshot=root/'history/football-csv/PL/2026.json'
            legacy=json.loads(snapshot.read_text());legacy.pop('archive_sha256');legacy['matches'][0]['score']['fullTime']['home']=9
            legacy['sha256']=digest({k:v for k,v in legacy.items() if k!='sha256'});snapshot.write_text(json.dumps(legacy))
            repaired=run(1,active_max_age_hours=6);self.assertTrue(repaired['complete']);self.assertEqual(calls.count(url),1)
            fixed=json.loads(snapshot.read_text());self.assertEqual(fixed['matches'][0]['score']['fullTime']['home'],2)
            self.assertEqual(fixed['fetched_at'],now.isoformat())
            saved=snapshot.read_bytes();run(5,active_max_age_hours=6);self.assertEqual(snapshot.read_bytes(),saved)
            payload=zipped(4)
            refreshed=run(6,active_max_age_hours=6);self.assertTrue(refreshed['complete']);self.assertEqual(calls.count(url),2)
            self.assertEqual(json.loads(snapshot.read_text())['matches'][0]['score']['fullTime']['home'],4)
            protected=[root/'archives/2026.zip',root/'archives/2026.receipt.json',snapshot]
            previous={p:p.read_bytes() for p in protected}
            payload=b'not a ZIP'
            failed=run(12,active_max_age_hours=6);self.assertFalse(failed['complete']);self.assertEqual(failed['collections'],[])
            for p,before in previous.items():self.assertEqual(p.read_bytes(),before)
            (root/'archives/2026.receipt.json').unlink()
            (root/'history/football-csv/FL1/2026.json').unlink()
            pinned=run(13);self.assertTrue(pinned['complete']);self.assertEqual(calls.count(url),3)
            unknown=json.loads((root/'history/football-csv/FL1/2026.json').read_text())
            self.assertIsNone(unknown['fetched_at']);self.assertEqual(unknown['normalized_at'],(now+timedelta(hours=13)).isoformat())
            collections,manifest=load_history(root/'history');self.assertEqual(len(collections),2)
            self.assertTrue(all(c['cache']['state']=='pinned' for c in manifest['collections']))
            self.assertEqual(calls.count(INDEX_URL),6)
            for response in responses:response.close.assert_called_once()

    def test_source_overlap_reports_score_conflicts_and_unmatched_games(self):
        from app.data_service.history import compare_sources

        left = [
            match(1, 2020),
            match(2, 2020, home="Liverpool"),
            match(3, 2020, home="Everton"),
        ]
        right = [
            match(1, 2020, home="Arsenal FC", away="Chelsea FC", source="openfootball"),
            match(
                2,
                2020,
                home="Liverpool FC",
                away="Chelsea FC",
                score=(1, 0),
                source="openfootball",
            ),
        ]
        report = compare_sources(left, right)
        self.assertEqual(report["matched"], 2)
        self.assertEqual(report["score_conflicts"], 1)
        self.assertEqual(report["only_left"], 1)
        self.assertEqual(report["score_agreement"], 0.5)
        import copy
        for csv_name, public_name in [('Troyes','ES Troyes AC'), ('Elversberg','SV 07 Elversberg'),
                                      ('Frosinone','Frosinone Calcio'), ('Santander','Real Racing Club de Santander'),
                                      ('La Coruna','RC Deportivo La Coruña')]:
            a, b = match(4,2026,home=csv_name), match(4,2026,home=public_name,source='openfootball')
            b['id'] += 100
            original=copy.deepcopy((a,b))
            compared=compare_sources([a],[b])
            self.assertEqual(compared['matched'],1)
            self.assertEqual((a,b),original)  # Comparison never reassigns provider identities.

    def test_ambiguous_club_changes_are_not_merged(self):
        from app.data_service.history import compare_sources

        left = [match(1, 2020, home="Belenenses")]
        right = [match(1, 2020, home="Belenenses SAD", source="openfootball")]
        report = compare_sources(left, right)
        self.assertEqual(report["matched"], 0)
        self.assertEqual(report["only_left"], 1)
        self.assertEqual(report["only_right"], 1)

    def test_year_conflicts_are_detected_and_isolated_without_changing_snapshots(self):
        from app.data_service.history import audit_history

        left = match(1, 2020)
        right = match(
            1, 2019, home="Arsenal FC", away="Chelsea FC", source="openfootball"
        )
        collections = {
            ("football-csv", "PL", "2019"): [left],
            ("openfootball", "PL", "2019"): [right],
        }
        audit = audit_history(collections)
        self.assertEqual(audit["comparisons"][0]["date_conflicts"], 1)
        self.assertEqual(len(audit["excluded_records"]), 2)
        self.assertEqual(right["source_date"], "2019-08-02")

    def test_all_score_conflicts_are_isolated_beyond_example_limit(self):
        from app.data_service.history import audit_history

        left = [match(i, 2020, home=f"Team {i}") for i in range(25)]
        right = [
            match(i, 2020, home=f"Team {i}", score=(0, 0), source="openfootball")
            for i in range(25)
        ]
        audit = audit_history(
            {
                ("football-csv", "PL", "2020"): left,
                ("openfootball", "PL", "2020"): right,
            }
        )
        self.assertEqual(audit["comparisons"][0]["score_conflicts"], 25)
        self.assertEqual(len(audit["excluded_records"]), 50)

    def test_failed_collection_is_not_an_empty_success(self):
        from app.data_service.history import collect_history

        client = Mock(source_name="football-data.co.uk")
        client.fetch_matches.side_effect = RuntimeError("offline")
        with tempfile.TemporaryDirectory() as directory:
            report = collect_history(
                Path(directory),
                ["football-csv"],
                ["PL"],
                ["2020"],
                client_factory=lambda _: client,
            )
        self.assertEqual(report["collections"], [])
        self.assertEqual(len(report["errors"]), 1)

    def test_experimental_identity_survives_provider_name_changes(self):
        from app.ml.seasonal import prepare_matches

        old = match(
            1, 2020, home="Arsenal FC", away="Chelsea FC", source="openfootball"
        )
        new = match(1, 2021, home="Arsenal", away="Chelsea", source="openfootball")
        prepared = prepare_matches([old, new], "openfootball", "PL")
        self.assertEqual(prepared[0]["homeTeam"]["id"], prepared[1]["homeTeam"]["id"])
        self.assertTrue(all(m["utcDate"].endswith("T00:00:00+00:00") for m in prepared))

    def test_season_holdout_and_pool_share_the_same_test_population(self):
        from app.ml.seasonal import run_season_matrix

        class UniformEstimator:
            fit_counts = []

            def fit(self, rows, targets):
                self.fit_count = len(rows)
                self.fit_counts.append(len(rows))

            def predict_proba(self, rows):
                return [[1 / 3] * 3 for _ in rows]

            def predict(self, rows):
                return [2 for _ in rows]

        collections = {}
        for code in ("PL", "SA"):
            for year in range(2020, 2023):
                collections[("football-csv", code, str(year))] = [
                    match(i, year, score=(i % 3, 1)) for i in range(40)
                ]
        report = run_season_matrix(
            collections,
            test_seasons=["2022"],
            windows=["1", "all"],
            models=["sgd"],
            factory=lambda *args, **kwargs: lambda _: UniformEstimator(),
        )
        self.assertEqual(len(report["results"]), 8)
        self.assertEqual(report["errors"], [])
        self.assertEqual(UniformEstimator.fit_counts, [80, 40, 40, 160, 80, 80])
        for row in report["results"]:
            self.assertNotIn("2022", row["training_seasons"])
            self.assertEqual(row["test_rows"], 40)
            self.assertLess(row["training_end"], row["test_start"])
        for code in ("PL", "SA"):
            subset = [r for r in report["results"] if r["competition"] == code]
            self.assertEqual(len({r["test_population_sha256"] for r in subset}), 1)


class TestMarketReference(unittest.TestCase):
    def test_three_way_odds_include_draw_and_reject_incomplete_or_invalid_prices(self):
        from app.analytics.market_reference import implied_probabilities

        probabilities, overround = implied_probabilities({"home": 2.0, "draw": 4.0, "away": 5.0})
        self.assertEqual(len(probabilities), 3)
        self.assertAlmostEqual(sum(probabilities), 1)
        self.assertAlmostEqual(probabilities[0], .2 / .95)  # away, draw, home
        self.assertAlmostEqual(probabilities[1], .25 / .95)
        self.assertAlmostEqual(probabilities[2], .5 / .95)
        self.assertAlmostEqual(overround, -.05)  # Do not clip negative overrounds.
        for bad in (None, 0, 1, True, float("inf"), float("nan"), "3.0"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                implied_probabilities({"home": 2.0, "draw": bad, "away": 5.0})

    def test_market_stages_use_identical_rows_and_count_quality_and_missing_exclusions(self):
        from app.analytics.market_reference import BOOKS, LEAGUES, evaluate

        def row(index, day, *, prices=True):
            r = match(index, int(day[:4])); r["source_date"] = day
            if prices:
                r["odds"] = {stage: {book: {"home": 2.0, "draw": 4.0, "away": 5.0}
                                     for book in BOOKS} for stage in ("pre_close", "closing")}
            return r

        old = [row(1, "2024-08-01"), row(2, "2025-07-24")]  # Second is too late to train.
        current = [row(3, "2025-07-22"), row(4, "2025-07-23"), row(5, "2025-07-24"), row(6, "2025-07-25", prices=False)]
        current[2]["odds"]["closing"]["bet365"]["draw"] = None
        snapshots = [{"provider": "football-data.co.uk", "competition": code, "season": str(year), "matches": rows}
                     for code in LEAGUES for year, rows in [(2024, old), (2025, current)]]
        results, predictions = evaluate(snapshots, (2025,))
        b = next(r for r in results if r["competition"] == "PL" and r["book"] == "bet365")
        p = next(r for r in results if r["competition"] == "PL" and r["book"] == "pinnacle")
        self.assertEqual(b["population"], 4)
        self.assertEqual(b["paired_matches"], 2)
        self.assertEqual(b["excluded"]["incomplete_closing"], 1)
        self.assertEqual(b["excluded"]["incomplete_both"], 1)
        self.assertEqual(p["paired_matches"], 1)
        self.assertEqual(p["excluded"]["provider_stale_warning"], 3)
        self.assertEqual(b["training_rows"], 1)
        self.assertLess(b["training_end"], b["test_start"])
        self.assertEqual(b["prior_probabilities"], [.25, .25, .5])
        self.assertAlmostEqual(b["metrics"]["pre_close"]["log_loss"], b["metrics"]["closing"]["log_loss"])
        self.assertEqual(len([r for r in predictions if r["competition"] == "PL" and r["book"] == "bet365"]), 2)
        old[1]["score"]["fullTime"]["away"] = 99
        changed, _ = evaluate(snapshots, (2025,))
        self.assertEqual(results, changed)  # A result unavailable at the boundary cannot change any score.
        current.append(dict(current[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate(snapshots, (2025,))


class TestHistoryCLI(unittest.TestCase):
    def test_history_cli_preserves_successful_snapshot_when_another_season_fails(self):
        matches = FootballCSVClient().parse_csv_text(
            "PL",
            "Date,HomeTeam,AwayTeam,FTHG,FTAG\n01/08/2025,A,B,2,1\n",
            "https://public.invalid/example.csv",
        )

        def fetch(client, code, *, season, status):
            if season == "2024":
                raise RuntimeError("offline season")
            return matches

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "output.json"
            with patch.object(FootballCSVClient, "fetch_matches", new=fetch):
                code = pipeline.main(
                    [
                        "collect-history",
                        "--competitions",
                        "PL",
                        "--sources",
                        "football-csv",
                        "--seasons",
                        "2024,2025",
                        "--cache-dir",
                        str(root / "cache"),
                        "--output",
                        str(destination),
                    ]
                )
            self.assertEqual(code, 1)
            report = json.loads(destination.read_text())
            self.assertEqual(len(report["collections"]), 1)
            self.assertEqual(len(report["errors"]), 1)
            snapshot = json.loads(
                (root / "cache" / report["collections"][0]["path"]).read_text()
            )
            self.assertEqual(
                snapshot["matches"][0]["score"]["fullTime"], {"home": 2, "away": 1}
            )

class TestPublicSources(unittest.TestCase):
    def test_csv_timezones_ids_and_scores(self):
        from app.data_service.fetch.public import FootballCSVClient

        response = Mock(status_code=200)
        response.text = "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG\nE0,10/08/2024,15:00,Arsenal,Chelsea,2,1\n"
        client = FootballCSVClient(session=Mock(get=Mock(return_value=response)))
        match = client.fetch_matches("PL", season="2024", status="FINISHED")[0]
        self.assertEqual(match["utcDate"], "2024-08-10T14:00:00+00:00")
        self.assertEqual(match["score"]["fullTime"], {"home": 2, "away": 1})
        self.assertEqual(
            match["id"], client.fetch_matches("PL", season="2024")[0]["id"]
        )
        self.assertNotEqual(match["homeTeam"]["id"], match["awayTeam"]["id"])
        self.assertNotIn(
            "X-Auth-Token", client.session.get.call_args.kwargs.get("headers", {})
        )

    def test_csv_rejects_bad_payload_and_unsupported_league(self):
        from app.data_service.fetch.public import FootballCSVClient
        from app.data_service.fetch.fetcher import DataSourceError

        response = Mock(status_code=200, text="<html>error</html>")
        client = FootballCSVClient(session=Mock(get=Mock(return_value=response)))
        for code in ["PL", "WC"]:
            with self.subTest(code=code), self.assertRaises(DataSourceError):
                client.fetch_matches(code, season="2024")

    def test_openfootball_date_only_fixtures_are_not_timed(self):
        from app.data_service.fetch.public import OpenFootballClient

        response = Mock(status_code=200)
        response.json.return_value = {
            "matches": [
                {
                    "date": "2024-08-10",
                    "team1": "Arsenal FC",
                    "team2": "Chelsea FC",
                    "score": {"ft": [2, 1]},
                },
                {"date": "2024-08-11", "team1": "Arsenal FC", "team2": "Chelsea FC"},
            ]
        }
        client = OpenFootballClient(session=Mock(get=Mock(return_value=response)))
        matches = client.fetch_matches("PL", season="2024")
        self.assertEqual(matches[0]["status"], "FINISHED")
        self.assertEqual(matches[1]["time_precision"], "date")
        self.assertEqual(
            client.fetch_scheduled_matches("PL", "2024-08-11", "2024-08-11"), []
        )

    def test_source_factory_without_key(self):
        from app.data_service.fetch.public import create_data_client

        with patch.dict(
            "os.environ",
            {"FOOTBALL_DATA_API_KEY": "", "SOCCER_ANALYTICS_DATA_SOURCE": "auto"},
        ):
            self.assertEqual(create_data_client().source_name, "football-data.co.uk")

    def test_openfootball_flat_scores_in_current_season(self):
        from app.data_service.fetch.public import OpenFootballClient

        response = Mock(status_code=200)
        response.json.return_value = {
            "matches": [
                {
                    "date": "2026-09-05",
                    "time": "15:00",
                    "team1": "Nottingham Forest FC",
                    "team2": "Tottenham Hotspur FC",
                    "score": [0, 0],
                }
            ]
        }
        matches = OpenFootballClient(
            session=Mock(get=Mock(return_value=response))
        ).fetch_matches("PL", season="2026")
        self.assertEqual(matches[0]["score"]["fullTime"], {"home": 0, "away": 0})
        self.assertEqual(matches[0]["status"], "FINISHED")

    def test_openfootball_pinned_rounds_partial_scores_and_response_cleanup(self):
        import copy
        from app.data_service.fetch.public import OpenFootballClient
        from app.data_service.fetch.fetcher import DataSourceError
        from app.ml.feature_engineering import FeatureEngineer

        fixtures = []
        for index, score in enumerate(({'ft':[2,1],'ht':[1,0],'et':[3,1],'p':[4,2]},
                                        {'ft':[0,0],'ht':None}, {'ft':[1,1],'ht':[True,0]},
                                        {'ft':[1,0],'ht':[2,0]}, [0,0])):
            fixtures.append({'round':f'Matchday {index+1}', 'date':f'2026-08-{index+1:02}',
                             'team1':'Arsenal', 'team2':'Chelsea', 'score':score})
        payload={'matches':fixtures}
        response=Mock(status_code=200);response.json.return_value=payload
        session=Mock(get=Mock(return_value=response))
        revision='ABCDEF0123456789ABCDEF0123456789ABCDEF01'
        client=OpenFootballClient(session=session,revision=revision)
        rows=client.fetch_matches('PL',season='2026')
        self.assertIn('/'+revision.lower()+'/2026-27/en.1.json',session.get.call_args.args[0])
        self.assertEqual(len(rows),5)
        self.assertEqual(rows[0]['round'],'Matchday 1')
        self.assertEqual(rows[0]['score']['fullTime'],{'home':2,'away':1})
        self.assertEqual(rows[0]['score']['halfTime'],{'home':1,'away':0})
        self.assertEqual(rows[0]['source_score']['et'],[3,1])
        self.assertEqual(rows[0]['source_score']['p'],[4,2])
        self.assertEqual([row.get('score_metadata',{}).get('half_time_status') for row in rows],
                         ['valid','missing','invalid','valid',None])
        self.assertFalse(rows[3]['score_metadata']['half_time_consistent_with_full_time'])
        self.assertEqual(rows[2]['source_score']['ht'],[True,0])
        self.assertNotIn('halfTime',rows[4]['score'])
        response.close.assert_called_once()
        old=copy.deepcopy(rows)
        for row in old:
            for key in ('round','source_score','score_metadata'):row.pop(key,None)
            row['score'].pop('halfTime',None)
        self.assertEqual(FeatureEngineer().build_training_dataset(rows),FeatureEngineer().build_training_dataset(old))
        for bad in ('master','../main','f'*39,True,123):
            with self.subTest(revision=bad), self.assertRaises(ValueError):OpenFootballClient(revision=bad)
        for broken in ({'matches':[dict(fixtures[0],score={'ft':[True,1]})]}, None):
            response.reset_mock()
            response.json.return_value=broken
            with self.assertRaises(DataSourceError):client.fetch_matches('PL',season='2026')
            response.close.assert_called_once()
        response.reset_mock();response.json.side_effect=ValueError('bad JSON')
        with self.assertRaises(DataSourceError):client.fetch_matches('PL',season='2026')
        response.close.assert_called_once()

    def test_public_schedule_spans_all_seasons_and_keeps_date_only_precision(self):
        from datetime import date
        from app.data_service.fetch.public import OpenFootballClient

        client=OpenFootballClient()
        def fixtures(code,*,season):
            day=date(int(season),8,1)
            return [client._match(code,day,clock,'A','B',scores,'Europe/London','source')
                    for clock,scores in [('15:00',None),(None,None),('17:00',(1,0))]]
        with patch.object(client,'fetch_matches',side_effect=fixtures) as fetch:
            rows=client.fetch_scheduled_matches('PL','2024-07-01','2027-06-30')
            self.assertEqual([c.kwargs['season'] for c in fetch.call_args_list],['2024','2025','2026'])
            self.assertEqual(len(rows),3)
            self.assertTrue(all(row['status']=='TIMED' and row['time_precision']=='minute' for row in rows))
            fetch.reset_mock()
            client.fetch_scheduled_matches('PL','2025-06-30','2025-07-01')
            self.assertEqual([c.kwargs['season'] for c in fetch.call_args_list],['2024','2025'])
            fetch.reset_mock()
            with self.assertRaises(ValueError):client.fetch_scheduled_matches('PL','2026-08-02','2026-08-01')
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
