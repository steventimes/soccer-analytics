import os
import unittest
from unittest.mock import patch

from app.config import DEFAULT_COMPETITIONS_MAP, DEFAULT_SEASONS, load_settings, resolve_competitions


class TestConfig(unittest.TestCase):
    def test_load_settings_uses_env_overrides(self):
        env = {
            "SOCCER_ANALYTICS_COMPETITIONS": "PL,SA",
            "SOCCER_ANALYTICS_TRAINING_SEASONS": "2022,2024",
            "SOCCER_ANALYTICS_PREDICTION_DAYS": "5",
            "SOCCER_ANALYTICS_SITE_EXPORT_DAYS": "2",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = load_settings()

        self.assertEqual(settings.competitions_map, {"PL": 2021, "SA": 2019})
        self.assertEqual(settings.training_seasons, ["2022", "2024"])
        self.assertEqual(settings.prediction_days, 5)
        self.assertEqual(settings.site_export_days, 2)

    def test_resolve_competitions_falls_back_when_filter_is_empty(self):
        settings = load_settings()
        resolved = resolve_competitions("UNKNOWN", settings)
        self.assertEqual(resolved, settings.competitions_map)

    def test_defaults_remain_available_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = load_settings()

        self.assertEqual(settings.competitions_map, DEFAULT_COMPETITIONS_MAP)
        self.assertEqual(settings.training_seasons, DEFAULT_SEASONS)
