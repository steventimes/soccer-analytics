"""QA-style CI contract tests.

Focus:
- Prevent regression of GitHub Pages enablement failures.
- Prevent pandas/Python runtime compatibility regressions.

The suite includes a property-based style test (generated input space)
so it can scale cleanly in larger codebases without brittle case-by-case
examples.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import Tuple


Version = Tuple[int, int, int]

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGES_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pages.yml"
DAILY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "daily_update.yml"
REQUIREMENTS = REPO_ROOT / "requirements.txt"


def parse_python_version_from_daily_workflow(contents: str) -> Tuple[int, int]:
    """Extract `python-version: "x.y"` from workflow text."""
    match = re.search(r'python-version:\s*"(?P<major>\d+)\.(?P<minor>\d+)"', contents)
    if not match:
        raise AssertionError("python-version not found in daily_update workflow")
    return int(match.group("major")), int(match.group("minor"))


def parse_pandas_pin(requirements_text: str) -> Version:
    """Extract `pandas==x.y.z` from requirements text."""
    match = re.search(r"^pandas==(?P<version>\d+\.\d+\.\d+)$", requirements_text, re.M)
    if not match:
        raise AssertionError("Expected strict pandas==x.y.z in requirements.txt")
    major, minor, patch = (int(part) for part in match.group("version").split("."))
    return major, minor, patch


def pandas_is_compatible_with_python(pandas: Version, python_mm: Tuple[int, int]) -> bool:
    """Compatibility rule used by QA contract tests.

    Current known constraint:
    - pandas >= 2.3.0 requires Python >= 3.11
    """
    if pandas >= (2, 3, 0) and python_mm < (3, 11):
        return False
    return True


class TestPagesWorkflowContract(unittest.TestCase):
    """Regression checks for GitHub Pages deployment configuration."""

    def test_configure_pages_includes_auto_enablement(self) -> None:
        contents = PAGES_WORKFLOW.read_text(encoding="utf-8")

        self.assertRegex(
            contents,
            r"uses:\s*actions/configure-pages@v5\n\s*with:\n\s*enablement:\s*true",
            "Expected configure-pages to include enablement: true for repos "
            "where Pages is not pre-enabled.",
        )


class TestDependencyCompatibilityContract(unittest.TestCase):
    """Contract checks for workflow/runtime dependency compatibility."""

    def test_repository_pin_matches_workflow_runtime(self) -> None:
        daily = DAILY_WORKFLOW.read_text(encoding="utf-8")
        requirements = REQUIREMENTS.read_text(encoding="utf-8")

        python_mm = parse_python_version_from_daily_workflow(daily)
        pandas_version = parse_pandas_pin(requirements)

        self.assertTrue(
            pandas_is_compatible_with_python(pandas_version, python_mm),
            f"Incompatible versions: pandas=={pandas_version[0]}.{pandas_version[1]}.{pandas_version[2]} "
            f"with python {python_mm[0]}.{python_mm[1]} in daily workflow.",
        )

    def test_property_python_310_rejects_all_pandas_2_3_or_higher(self) -> None:
        """Property-based invariant over generated versions.

        Instead of one fixed example, this verifies a whole range of generated
        pandas patch/minor combinations to reduce blind spots.
        """
        python_mm = (3, 10)
        # Generated input space (property-style): all 2.3.x..2.6.x samples.
        for minor in range(3, 7):
            for patch in range(0, 11):
                pandas_version = (2, minor, patch)
                with self.subTest(pandas=pandas_version):
                    self.assertFalse(
                        pandas_is_compatible_with_python(pandas_version, python_mm),
                        "pandas >=2.3.0 must be rejected for Python 3.10",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
