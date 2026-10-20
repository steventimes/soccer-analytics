"""Shared execution for independent checks grouped under one unittest test."""

import importlib.util
import unittest


class ContractTestCase(unittest.TestCase):
    def _run_contracts(self, names: tuple[str, ...]) -> None:
        for name in names:
            with self.subTest(contract=name):
                getattr(self, name)()


def requires_modules(*names):
    """Skip only checks that execute a specific optional runtime."""
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    return unittest.skipIf(bool(missing), "Optional dependencies missing: " + ", ".join(missing))
