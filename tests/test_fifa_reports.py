import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from app.data_service import fifa_reports
from app.data_service.fifa_reports import parse_report, report_links, shape_metrics


class TestFifaReports(unittest.TestCase):
    def offers_fixture(self):
        header = "# Player Total Offers In Front In Between Out to In In to Out In Behind No Movement Offers Received\n"
        pages = [
            "13 July 2025 - Stadium - 15:00\nOffering to Receive Alpha FC\n 10 4\nTotal Offers Made Total Offers Received\n",
            "Offering to Receive Beta FC\n 5 2\nTotal Offers Made Total Offers Received\n",
            "Divider\n",
            "In Possession - Offers & Receptions Alpha FC\n" + header + "1 Alpha PLAYER 10 2 3 0 0 1 4 4\n",
            "In Possession - Offers & Receptions Beta FC\n" + header + "9 Beta PLAYER 5 0 2 1 0 2 0 2\n",
        ]
        rosters = {0: {1: "Alpha PLAYER"}, 1: {9: "Beta PLAYER"}}
        return pages, rosters

    def test_offers_reconcile_categories_rosters_and_separate_overview_pages(self):
        from app.analytics.fifa_offers import parse_offers

        pages, rosters = self.offers_fixture()
        rows, teams = parse_offers(pages, ["Alpha FC", "Beta FC"], rosters)
        self.assertEqual(teams[0]["total_offers"], 10)
        self.assertEqual(teams[1]["offers_received"], 2)
        self.assertEqual(teams[0]["player_table_page"], 4)
        self.assertEqual(teams[0]["overview_page"], 1)
        self.assertEqual(rows[0]["out_to_in"], 0)
        self.assertEqual(rows[0]["player_name"], "Alpha PLAYER")
        self.assertEqual(teams[0]["players"], 1)

    def test_offers_reject_missing_cells_wrong_identities_and_unreconciled_counts(self):
        from app.analytics.fifa_offers import parse_offers

        pages, rosters = self.offers_fixture()
        changes = [
            (3, pages[3].replace("10 2 3 0 0 1 4 4", "10 2 3 0 1 4 4")),
            (3, pages[3].replace("10 2 3", "11 2 3")),
            (3, pages[3].replace("Alpha FC", "Other FC")),
            (3, pages[3].replace("Alpha PLAYER", "Other PLAYER")),
            (3, pages[3] + "1 Alpha PLAYER 10 2 3 0 0 1 4 4\n"),
            (3, pages[3].replace("1 Alpha PLAYER", "2 Alpha PLAYER")),
            (3, pages[3].replace("Offers Received", "Other Meaning")),
            (0, pages[0].replace("10 4", "10 5")),
            (3, "Divider"),
        ]
        for index, value in changes:
            with self.subTest(index=index, value=value):
                changed = pages.copy(); changed[index] = value
                with self.assertRaises(ValueError):
                    parse_offers(changed, ["Alpha FC", "Beta FC"], rosters)

    def test_physical_zones_preserve_provider_labels_and_reject_inconsistent_totals(
        self,
    ):
        text = "Physical Data Alpha FC\nZone 1: 0-7 km/h Zone 2: 7-15 km/h Zone 3: 15-20 km/h Zone 4: 20-25 km/h Zone 5: 25+ km/h High Speed Runs Sprints Top Speed\n(Zone 3) (Zone 4 & 5)\n10 Player NAME 10000.0 3000.0 4000.0 2000.0 700.0 300.0 150.0 45.0 33.1\n"
        rows = fifa_reports.physical_rows([text], ["Alpha FC", "Beta FC"])
        self.assertEqual(rows[0]["zone_5_above_25_kmh_m"], 300)
        self.assertEqual(rows[0]["sprints_zone_4_5_count"], 45)
        earlier = fifa_reports.physical_rows(
            [text.replace("(Zone 3) (Zone 4 & 5)", "(Zone 4) (Zone 5)")],
            ["Alpha FC", "Beta FC"],
        )[0]
        self.assertEqual(earlier["sprints_zone_5_count"], 45)
        self.assertNotIn("sprints_zone_4_5_count", earlier)
        self.assertEqual(rows[0]["player_name"], "Player NAME")
        for changed in [
            text.replace("10000.0", "11000.0"),
            text.replace("Alpha FC", "Other FC"),
            text.replace("25+", "24+"),
        ]:
            with self.assertRaises(ValueError):
                fifa_reports.physical_rows([changed], ["Alpha FC", "Beta FC"])

    def test_official_collection_limits_requests_and_reuses_verified_cache(self):
        settings = {
            "edition": 2025,
            "matches": 1,
            "competition": "Test",
            "indices": ["https://www.fifatrainingcentre.com/hub.php"],
        }

        def response(url, **kwargs):
            body = (
                b"%PDF-1.7\npublic-test"
                if url.endswith(".pdf")
                else b'<a href="/media/native/M01 A V B.pdf">Report</a>'
            )
            result = MagicMock()
            result.__enter__.return_value = result
            result.iter_content.return_value = iter([body])
            return result

        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            fifa_reports.SERIES, {"test": settings}
        ), patch.object(fifa_reports.requests, "Session") as session, patch.object(
            fifa_reports, "SlidingWindowRateLimiter"
        ) as limiter:
            session.return_value.__enter__.return_value.get.side_effect = response
            manifest = fifa_reports.collect(temporary, "test")
            self.assertEqual(manifest["errors"], [])
            self.assertEqual(len(manifest["files"]), 2)
            self.assertTrue(
                (Path(temporary) / "test" / "M001.pdf")
                .read_bytes()
                .startswith(b"%PDF-")
            )
            self.assertEqual(limiter.return_value.add_request.call_count, 2)
            self.assertEqual(limiter.return_value.wait_if_needed.call_count, 2)
            session.reset_mock()
            limiter.return_value.reset_mock()
            fifa_reports.collect(temporary, "test")
            session.assert_not_called()
            limiter.return_value.add_request.assert_not_called()

    def valid_pages(self):
        return [
            "Alpha FC 3 - 0 Beta FC\nFinal - Match 63\n13 July 2025\n",
            "teams",
            "Total 32.3% 6.7% 61% Total\n3 Goals 0\n9(5) Attempts at Goal (On Target) 9(6)\n327(263) Total Passes (Complete) 620(567)\n287(44) Defensive Pressures Applied (Direct Pressures) 199(49)\n",
            "9% Long Ball 1%\n6% High Press 10%\n",
        ]

    def test_report_retains_three_state_possession_and_team_sides(self):
        metadata, rows = parse_report(self.valid_pages(), 63, 2025)
        self.assertEqual(metadata["goals"], [3, 0])
        self.assertEqual(metadata["date"], "2025-07-13")
        self.assertEqual(metadata["in_contest_pct"], 6.7)
        values = {(r["team_slot"], r["metric"]): r["value"] for r in rows}
        self.assertEqual(values[0, "controlled_possession_pct"], 32.3)
        self.assertEqual(values[1, "controlled_possession_pct"], 61)
        self.assertEqual(values[0, "direct_pressures"], 44)
        self.assertEqual(values[1, "in_possession_long_ball"], 1)
        self.assertNotIn((0, "ball_recovery_time"), values)

    def test_wrapped_cover_uses_explicit_team_summary_sides(self):
        pages = self.valid_pages()
        pages[0] = (
            "Alpha       3-   Beta\nFC          0    FC\nFinal - Match 63\n13 July 2025\n"
        )
        pages[1] = "Match Summary - Teams\nAlpha FC   3     0    Beta FC\nSTARTING\n"
        parsed, _ = parse_report(pages, 63, 2025)
        self.assertEqual(parsed["teams"], ["Alpha FC", "Beta FC"])
        self.assertEqual(parsed["goals"], [3, 0])
        self.assertEqual(parsed["stage"], "Final")

    def test_wrong_identity_edition_or_missing_core_metrics_is_rejected(self):
        for number, edition in [(62, 2025), (63, 2026)]:
            with self.assertRaises(ValueError):
                parse_report(self.valid_pages(), number, edition)
        p = self.valid_pages()
        p[2] = ""
        with self.assertRaises(ValueError):
            parse_report(p, 63, 2025)

    def test_index_accepts_official_practice_paths_and_rejects_other_hosts(self):
        html = '<a href="/media/native/practice/M61 FLU V CHE.pdf">61</a><a href="https://evil.invalid/M63 CHE V PSG.pdf">63</a><a href="/media/native/M64 invalid.pdf">bad</a>'
        with self.assertRaises(ValueError):
            report_links(html, "https://www.fifatrainingcentre.com/hub.php", 63)
        self.assertEqual(
            set(
                report_links(
                    html.split('<a href="/media/native/M64')[0],
                    "https://www.fifatrainingcentre.com/hub.php",
                    63,
                )
            ),
            {61},
        )

    def test_dimensions_follow_annotation_position_not_text_reading_order(self):
        words = []
        for offset, values in [
            (0, [34, 43, 48]),
            (300, [26, 42, 40]),
            (600, [18, 36, 19]),
        ]:
            for x, v in zip([40, 150, 270], values):
                words.append(
                    f'<word xMin="{offset+x}" xMax="{offset+x+10}">{v}m</word>'
                )
        xml = (
            '<html xmlns="http://www.w3.org/1999/xhtml"><body><page width="900">'
            + "".join(reversed(words))
            + "</page></body></html>"
        )
        rows = shape_metrics(
            xml,
            ["Defensive Line Height & Team Length Alpha FC"],
            ["Alpha FC", "Beta FC"],
        )
        values = {r["metric"]: r["value"] for r in rows}
        self.assertEqual(values["defense_high_block_press_team_width"], 43)
        self.assertEqual(values["defense_high_block_press_line_height"], 48)
        self.assertEqual(values["defense_low_block_team_length"], 18)
