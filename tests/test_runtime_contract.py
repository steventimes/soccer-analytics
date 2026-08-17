from __future__ import annotations

import importlib
import math
import random
import re
import stat
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.ml.feature_engineering import FeatureEngineer
from app.ml.model import XGBoostClassifier


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "app"
OBSOLETE_MODULE_FILES = (
    "legacy_removed.py",
    "data_service/db_test.py",
    "data_service/db_session.py",
    "data_service/fetch/understat_client.py",
    "data_service/db/cache/cache_management.py",
    "data_service/db/data_service.py",
    "data_service/db/database/db_schema.py",
    "data_service/db/repositories/competition_repository.py",
    "data_service/db/repositories/match_repository.py",
    "data_service/db/repositories/team_repository.py",
    "seeds/seed_competitions.py",
    "seeds/seed_matches.py",
    "seeds/seed_players.py",
    "seeds/seed_real_odds.py",
    "seeds/seed_understat.py",
    "ml/predict_upcoming.py",
    "ml/simulate_betting.py",
    "run/run_betting_simulation.py",
    "run/run_predictions.py",
    "run/run_training.py",
    "run_betting_simulation.py",
    "run_predictions.py",
    "run_training.py",
    "web/daily_update.py",
    "web/predictions.py",
)


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


def _random_matches(seed: int, count: int = 90) -> list[dict]:
    rng = random.Random(seed)
    teams = list(range(1, 11))
    start = datetime(2025, 1, 1, 12, tzinfo=UTC)
    matches = []
    for index in range(count):
        home, away = rng.sample(teams, 2)
        matches.append(
            {
                "id": index + 1,
                "utcDate": (start + timedelta(days=index)).isoformat(),
                "status": "FINISHED",
                "homeTeam": {"id": home},
                "awayTeam": {"id": away},
                "score": {
                    "fullTime": {
                        "home": rng.randrange(0, 5),
                        "away": rng.randrange(0, 5),
                    }
                },
            }
        )
    return matches


class TestRuntimeContract(unittest.TestCase):
    def test_repository_surface_has_only_current_importable_modules(self):
        remaining = [
            path for path in OBSOLETE_MODULE_FILES if (APP_ROOT / path).exists()
        ]
        self.assertEqual(remaining, [])

        failures = []
        for path in sorted(APP_ROOT.rglob("*.py")):
            try:
                importlib.import_module(_module_name(path))
            except Exception as exc:
                failures.append(
                    f"{_module_name(path)}: {type(exc).__name__}: {exc}"
                )
        self.assertEqual(failures, [], "\n".join(failures))

        requirements = [
            line.strip()
            for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(
            requirements,
            ["requests==2.32.5", "xgboost-cpu==3.4.1"],
        )

    def test_execution_surfaces_share_python_and_the_canonical_pipeline(self):
        workflow = (ROOT / ".github/workflows/daily_update.yml").read_text()
        dockerfile = (ROOT / "Dockerfile").read_text()
        runner_path = ROOT / "run.sh"
        runner = runner_path.read_text(encoding="utf-8")

        self.assertEqual(
            _python_version(workflow, r'python-version:\s*"(\d+)\.(\d+)"'),
            (3, 12),
        )
        self.assertEqual(
            _python_version(
                dockerfile,
                r"^FROM python:(\d+)\.(\d+)(?:[.-])",
            ),
            (3, 12),
        )
        self.assertIn("FOOTBALL_DATA_API_KEY", workflow)
        self.assertNotIn("DATABASE_URL", workflow)
        self.assertNotIn("REDIS_URL", workflow)
        self.assertIn("python -m app.pipeline all", workflow)
        self.assertNotIn("app.web.daily_update", workflow)
        self.assertTrue(runner_path.stat().st_mode & stat.S_IXUSR)
        self.assertIn("docker compose version", runner)
        self.assertLess(
            runner.index("docker compose version"),
            runner.index("docker-compose"),
        )

    def test_randomized_feature_and_model_invariants(self):
        engineer = FeatureEngineer()
        for seed in range(10):
            with self.subTest(seed=seed):
                matches = _random_matches(seed)
                dataset = engineer.build_training_dataset(matches)
                self.assertEqual(len(dataset.rows), 90)
                self.assertEqual(tuple(dataset.rows[0]), engineer.features)
                self.assertTrue(
                    all(
                        math.isfinite(value)
                        for row in dataset.rows
                        for value in row.values()
                    )
                )

        matches = _random_matches(42, 30)
        self.assertEqual(
            engineer.build_training_dataset(matches),
            engineer.build_training_dataset(reversed(matches)),
        )

        dataset = engineer.build_training_dataset(_random_matches(7, 120))
        model = XGBoostClassifier(engineer.features, rounds=40).fit(
            dataset.rows[:100],
            dataset.targets[:100],
        )
        self.assertTrue(
            all(
                math.isclose(sum(row), 1.0, abs_tol=1e-6)
                and all(
                    math.isfinite(value) and 0 <= value <= 1
                    for value in row
                )
                for row in model.predict_proba(dataset.rows[100:])
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
