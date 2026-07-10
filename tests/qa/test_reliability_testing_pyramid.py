"""Long-horizon reliability suite organized as a testing pyramid."""

from __future__ import annotations

import ast
import inspect
import os
import random
import re
import textwrap
import trace
import unittest
from datetime import date, timedelta
from itertools import product
from pathlib import Path
from typing import Sequence, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover - environment-dependent import guard
    pd = None  # type: ignore[assignment]



Version = Tuple[int, int, int]
RESULT_PAIRS = (("W", "L"), ("D", "D"), ("L", "W"))
REPO_ROOT = Path(__file__).resolve().parents[2]
PAGES_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pages.yml"
DAILY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "daily_update.yml"
REQUIREMENTS = REPO_ROOT / "requirements.txt"


def parse_python_version_from_daily_workflow(contents: str) -> Tuple[int, int]:
    match = re.search(r'python-version:\s*"(?P<major>\d+)\.(?P<minor>\d+)"', contents)
    if not match:
        raise AssertionError("python-version not found in daily_update workflow")
    return int(match.group("major")), int(match.group("minor"))


def parse_pandas_pin(requirements_text: str) -> Version:
    match = re.search(r"^pandas==(?P<version>\d+\.\d+\.\d+)$", requirements_text, re.M)
    if not match:
        raise AssertionError("Expected strict pandas==x.y.z in requirements.txt")
    return tuple(int(part) for part in match.group("version").split("."))  # type: ignore[return-value]


def pandas_is_compatible_with_python(pandas: Version, python_mm: Tuple[int, int]) -> bool:
    if pandas >= (2, 3, 0) and python_mm < (3, 11):
        return False
    return True


def _result_to_score(result: str) -> float:
    if result == "W":
        return 1.0
    if result == "D":
        return 0.5
    return 0.0


def _normalize_path(path: str) -> str:
    try:
        return os.path.normcase(os.path.normpath(str(Path(path).resolve())))
    except OSError:
        return os.path.normcase(os.path.normpath(path))


def _statement_lines(fn: object) -> set[int]:
    source_lines, start = inspect.getsourcelines(fn)
    tree = ast.parse(textwrap.dedent("".join(source_lines)))
    fn_node = tree.body[0]
    if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    return {
        start + node.lineno - 1
        for node in ast.walk(fn_node)
        if isinstance(node, ast.stmt) and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _build_two_team_fixture(outcomes: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    if pd is None:
        raise RuntimeError("pandas is required for fixture generation")
    rows: list[dict[str, object]] = []
    base_date = date(2025, 1, 1)
    for idx, (a_result, b_result) in enumerate(outcomes, start=1):
        match_date = (base_date + timedelta(days=idx)).isoformat()
        rows.append(
            {
                "id": idx,
                "date": match_date,
                "teamID": "A",
                "opponentID": "B",
                "result": a_result,
                "location": "h",
                "xGoals": 1.7 if a_result == "W" else 1.2 if a_result == "D" else 0.8,
                "deep": 9 if a_result == "W" else 7 if a_result == "D" else 5,
                "ppda": 7.0 if a_result == "W" else 9.0 if a_result == "D" else 11.0,
                "goals": 2 if a_result == "W" else 1 if a_result == "D" else 0,
            }
        )
        rows.append(
            {
                "id": idx,
                "date": match_date,
                "teamID": "B",
                "opponentID": "A",
                "result": b_result,
                "location": "a",
                "xGoals": 1.7 if b_result == "W" else 1.2 if b_result == "D" else 0.8,
                "deep": 9 if b_result == "W" else 7 if b_result == "D" else 5,
                "ppda": 7.0 if b_result == "W" else 9.0 if b_result == "D" else 11.0,
                "goals": 2 if b_result == "W" else 1 if b_result == "D" else 0,
            }
        )
    return pd.DataFrame(rows)


def _build_random_fixture(seed: int, match_count: int = 24) -> pd.DataFrame:
    if pd is None:
        raise RuntimeError("pandas is required for fixture generation")
    rng = random.Random(seed)
    team_pool = ["A", "B", "C", "D", "E", "F", "G", "H"]
    rows: list[dict[str, object]] = []
    base_date = date(2025, 1, 1)
    for idx in range(1, match_count + 1):
        t1, t2 = rng.sample(team_pool, 2)
        r1, r2 = rng.choice(RESULT_PAIRS)
        match_date = (base_date + timedelta(days=idx)).isoformat()
        xg1 = round(rng.uniform(0.2, 2.8), 3)
        xg2 = round(rng.uniform(0.2, 2.8), 3)
        deep1 = rng.randint(1, 20)
        deep2 = rng.randint(1, 20)
        ppda1 = round(rng.uniform(4.0, 22.0), 3)
        ppda2 = round(rng.uniform(4.0, 22.0), 3)
        goals1 = rng.randint(0, 4)
        goals2 = rng.randint(0, 4)
        rows.append(
            {
                "id": idx,
                "date": match_date,
                "teamID": t1,
                "opponentID": t2,
                "result": r1,
                "location": "h",
                "xGoals": xg1,
                "deep": deep1,
                "ppda": ppda1,
                "goals": goals1,
            }
        )
        rows.append(
            {
                "id": idx,
                "date": match_date,
                "teamID": t2,
                "opponentID": t1,
                "result": r2,
                "location": "a",
                "xGoals": xg2,
                "deep": deep2,
                "ppda": ppda2,
                "goals": goals2,
            }
        )
    return pd.DataFrame(rows)


@unittest.skipIf(pd is None, "pandas is not installed")
class Test01SmokeSanity(unittest.TestCase):
    def test_core_components_bootstrap(self) -> None:
        from app.ml.feature_engineering import FeatureEngineer

        fe = FeatureEngineer()
        self.assertTrue(PAGES_WORKFLOW.exists())
        self.assertTrue(DAILY_WORKFLOW.exists())
        self.assertTrue(REQUIREMENTS.exists())
        self.assertTrue(fe.features)
        out = fe.calculate_rolling_features(_build_two_team_fixture([("W", "L"), ("D", "D"), ("L", "W")]))
        self.assertFalse(out.empty)


class Test02UnitContracts(unittest.TestCase):
    def test_dependency_contract(self) -> None:
        daily = DAILY_WORKFLOW.read_text(encoding="utf-8")
        requirements = REQUIREMENTS.read_text(encoding="utf-8")
        python_mm = parse_python_version_from_daily_workflow(daily)
        pandas_version = parse_pandas_pin(requirements)
        self.assertTrue(
            pandas_is_compatible_with_python(pandas_version, python_mm),
            f"Incompatible versions: pandas=={pandas_version} with python {python_mm}.",
        )

    def test_pages_enablement_contract(self) -> None:
        contents = PAGES_WORKFLOW.read_text(encoding="utf-8")
        self.assertRegex(
            contents,
            r"uses:\s*actions/configure-pages@v5\n\s*with:\n\s*enablement:\s*true",
            "Expected configure-pages to include enablement: true.",
        )


@unittest.skipIf(pd is None, "pandas is not installed")
class Test03Fuzzing(unittest.TestCase):
    def test_feature_engineering_fuzz_no_crash(self) -> None:
        from app.ml.feature_engineering import FeatureEngineer

        fe = FeatureEngineer()
        for seed in range(10):
            with self.subTest(seed=seed):
                df = _build_random_fixture(seed, match_count=24)
                out = fe.calculate_rolling_features(df)
                self.assertIn("elo_diff", out.columns)
                self.assertIn("team_elo", out.columns)
                self.assertFalse(out.empty)


@unittest.skipIf(pd is None, "pandas is not installed")
class Test04PropertyBasedTesting(unittest.TestCase):
    def test_mirrored_rows_have_mirrored_pre_match_elo(self) -> None:
        from app.ml.feature_engineering import FeatureEngineer

        fe = FeatureEngineer()
        for length in [2, 3, 4, 5]:
            for seq in product(RESULT_PAIRS, repeat=length):
                df = _build_two_team_fixture(seq)
                out = fe.calculate_rolling_features(df)
                grouped = out.groupby("id")
                for _, group in grouped:
                    if len(group) != 2:
                        continue
                    a = group.iloc[0]
                    b = group.iloc[1]
                    self.assertAlmostEqual(a["team_elo"], b["opp_elo"], places=10)
                    self.assertAlmostEqual(a["opp_elo"], b["team_elo"], places=10)


@unittest.skipIf(pd is None, "pandas is not installed")
class Test05Integration(unittest.TestCase):
    def test_pipeline_like_transformation_produces_training_ready_columns(self) -> None:
        from app.ml.feature_engineering import FeatureEngineer

        fe = FeatureEngineer()
        df = _build_two_team_fixture([("W", "L"), ("D", "D"), ("L", "W"), ("W", "L"), ("D", "D")])
        out = fe.calculate_rolling_features(df)

        expected_columns = set(fe.features) | {"target", "id", "teamID", "opponentID", "result"}
        missing = expected_columns - set(out.columns)
        self.assertFalse(missing, f"Missing expected columns: {sorted(missing)}")
        self.assertFalse(out[list(fe.features)].isna().any().any(), "Feature set should not include NaN values.")


@unittest.skipIf(pd is None, "pandas is not installed")
class Test06CoverageChecking(unittest.TestCase):
    def test_feature_engineering_core_path_has_minimum_line_coverage(self) -> None:
        from app.ml.feature_engineering import FeatureEngineer

        fe = FeatureEngineer()
        df = _build_two_team_fixture([("W", "L"), ("D", "D"), ("L", "W"), ("W", "L"), ("D", "D")])

        tracer = trace.Trace(count=True, trace=False)
        tracer.runfunc(fe.calculate_rolling_features, df)
        results = tracer.results()

        executable = _statement_lines(FeatureEngineer.calculate_rolling_features)
        module_path = _normalize_path(str(Path(inspect.getsourcefile(FeatureEngineer)).resolve()))
        hit = {
            lineno
            for (path, lineno), count in results.counts.items()
            if count > 0 and _normalize_path(path) == module_path
        }
        ratio = len(executable & hit) / max(len(executable), 1)
        self.assertGreaterEqual(ratio, 0.50, f"Coverage ratio too low: {ratio:.2%}")


class Test07ModelChecking(unittest.TestCase):
    def test_exhaustive_small_state_space_for_compatibility_rule(self) -> None:
        py_space = [(3, 9), (3, 10), (3, 11), (3, 12)]
        pandas_space: list[Version] = [(2, minor, patch) for minor in range(1, 5) for patch in range(0, 6)]
        for python_mm in py_space:
            for pandas_version in pandas_space:
                expected = not (pandas_version >= (2, 3, 0) and python_mm < (3, 11))
                with self.subTest(python_mm=python_mm, pandas=pandas_version):
                    self.assertEqual(
                        pandas_is_compatible_with_python(pandas_version, python_mm),
                        expected,
                    )


@unittest.skipIf(pd is None, "pandas is not installed")
class Test08TheoremLikeInvariant(unittest.TestCase):
    def test_elo_system_is_zero_sum_over_all_three_match_outcomes(self) -> None:
        from app.ml.feature_engineering import FeatureEngineer

        fe = FeatureEngineer()
        for length in [1, 2, 3, 4]:
            for seq in product(RESULT_PAIRS, repeat=length):
                df = _build_two_team_fixture(seq)
                out = fe.calculate_rolling_features(df)
                a_view = out[out["teamID"] == "A"].sort_values("date")

                ra = 1500.0
                rb = 1500.0
                for _, row in a_view.iterrows():
                    self.assertAlmostEqual(float(row["team_elo"]), ra, places=10)
                    self.assertAlmostEqual(float(row["opp_elo"]), rb, places=10)
                    score = _result_to_score(str(row["result"]))
                    expected = 1 / (1 + 10 ** ((rb - ra) / 400))
                    delta = 20 * (score - expected)
                    ra += delta
                    rb -= delta

                self.assertAlmostEqual(ra + rb, 3000.0, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
