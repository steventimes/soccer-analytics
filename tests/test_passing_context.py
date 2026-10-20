import csv
import json
from pathlib import Path
import tempfile
import unittest

from app.analytics.passing_context import (
    build_pass_contexts,
    summarize_passes,
    summarize_pressing_chains,
    run,
)


def event(event_id, event_type, **fields):
    return {
        "event_id": event_id,
        "event_type": event_type,
        "match_id": "1",
        "period": "1",
        "team_id": "10",
        "frame_start": "10",
        "frame_end": "20",
        **fields,
    }


def possession(**fields):
    return event("p", "player_possession", end_type="pass", **fields)


def option(event_id="o", **fields):
    return event(
        event_id,
        "passing_option",
        **{"associated_player_possession_event_id": "p",
           "passing_option_at_pass_moment": "True", **fields},
    )


def chain(event_id, **fields):
    return event(
        event_id,
        "on_ball_engagement",
        **{"pressing_chain_index": "1", "pressing_chain": "True", **fields},
    )


class TestPassingContext(unittest.TestCase):
    def test_selected_receiver_unavailable_is_preserved_outside_denominator(self):
        rows = [
            possession(targeted_passing_option_event_id="chosen"),
            option("available"),
            option("chosen", passing_option_at_pass_moment="False", targeted="True"),
            option("missing", passing_option_at_pass_moment=""),
        ]
        cases, joins = build_pass_contexts(rows, "1")
        case = cases[0]
        self.assertEqual(case["release_option_ids"], ["available"])
        self.assertEqual(case["release_options"], 1)
        self.assertEqual(case["selected_receiver_availability"], "false")
        self.assertTrue(case["selected_receiver_unavailable"])
        self.assertEqual(joins["passing_options_joined"], 3)
        self.assertEqual(summarize_passes(cases)["release_available_options"], 1)

    def test_flag_defines_release_denominator_interval_discrepancy_is_flagged(self):
        rows = [possession(), option(frame_end="18", peak_passing_option_frame="18")]
        cases, joins = build_pass_contexts(rows, "1")
        self.assertEqual(cases[0]["release_options"], 1)
        self.assertEqual(cases[0]["release_flag_outside_option_interval"], 1)
        self.assertEqual(joins["passing_options_joined"], 1)

    def test_join_rejects_cross_period_team_future_interval_and_missing_reference(self):
        rows = [
            possession(), option("ok"), option("period", period="2"),
            option("team", team_id="20"), option("future", frame_start="21", frame_end="30"),
            option("orphan", associated_player_possession_event_id="absent"),
        ]
        cases, joins = build_pass_contexts(rows, "1")
        self.assertEqual(cases[0]["release_option_ids"], ["ok"])
        self.assertEqual(joins["passing_options_joined"], 1)
        self.assertEqual(sum(joins["rejected_option_reasons"].values()), 4)

    def test_unknown_outcome_quality_and_raw_coordinates_are_not_zero_or_clipped(self):
        cases, _ = build_pass_contexts([possession(x_end="54", y_end="-35"), option()], "1")
        case = cases[0]
        self.assertEqual(case["coordinates"]["x_end"], 54)
        self.assertEqual(case["quality"]["fully_extrapolated"], "unknown")
        summary = summarize_passes(cases)
        self.assertEqual(summary["outcomes"], {"unknown": 1})
        self.assertIsNone(summary["completion_rate_known"])
        self.assertEqual(summary["quality"]["is_player_possession_end_matched"], {"unknown": 1})

    def test_duplicate_chain_members_never_inflate_regain_or_joined_value(self):
        pp = possession(possession_epv_delta_for="0.2", possession_epv_delta_against="0.1")
        rows = [pp] + [chain(str(i), pressing_chain_end_type="regain", team_id="20",
                             associated_player_possession_event_id="p") for i in range(5)]
        summary, records = summarize_pressing_chains(rows, "1")
        self.assertEqual(summary["chains"], 1)
        self.assertEqual(summary["outcomes"], {"regain": 1})
        self.assertEqual(records[0]["associated_possessions"], 1)
        self.assertEqual(records[0]["opponent_supplier_epv"]["possession_epv_delta_for"]["n"], 1)
        self.assertEqual(summary["opponent_supplier_epv_chain_means"]["possession_epv_delta_for"]["n"], 1)

    def test_chain_epv_average_weights_chains_after_deduplicating_spells(self):
        rows = [possession(possession_epv_delta_for="0.2"),
                dict(possession(possession_epv_delta_for="0.8"), event_id="q"),
                dict(possession(possession_epv_delta_for="0.2"), event_id="r")]
        rows += [chain("a", team_id="20", associated_player_possession_event_id="p"),
                 chain("b", team_id="20", associated_player_possession_event_id="q"),
                 chain("duplicate", team_id="20", associated_player_possession_event_id="q"),
                 chain("c", team_id="20", pressing_chain_index="2", associated_player_possession_event_id="r")]
        summary, _ = summarize_pressing_chains(rows, "1")
        value = summary["opponent_supplier_epv_chain_means"]["possession_epv_delta_for"]
        self.assertAlmostEqual(value["mean"], 0.35)
        self.assertEqual(value["n"], 2)
        self.assertNotIn("sum", value)
        self.assertEqual(summary["associated_possession_links"], 3)

    def test_period_team_and_match_chain_collisions_are_scoped(self):
        rows = [chain("a", pressing_chain_end_type="regain"),
                chain("b", period="2", pressing_chain_end_type="disruption"),
                chain("c", team_id="20")]
        summary, records = summarize_pressing_chains(rows, "1")
        self.assertEqual(summary["chains"], 3)
        self.assertEqual(summary["outcomes"], {"disruption": 1, "regain": 1, "unknown": 1})
        other, _ = summarize_pressing_chains([dict(row, match_id="2") for row in rows], "2")
        self.assertEqual(other["chains"], 3)
        self.assertEqual({r["match_id"] for r in records}, {"1"})

    def test_conflicting_and_missing_chain_outcomes_are_explicit(self):
        rows = [chain("a", pressing_chain_end_type="regain"),
                chain("b", pressing_chain_end_type="disruption"),
                chain("c", pressing_chain_index="2"),
                chain("d", pressing_chain_index="3", pressing_chain_end_type="regain")]
        summary, records = summarize_pressing_chains(rows, "1")
        self.assertEqual(summary["outcomes"], {"conflict": 1, "regain": 1, "unknown": 1})
        self.assertEqual(summary["known_outcome_chains"], 1)
        self.assertEqual(summary["regain_rate_known"], 1)
        self.assertEqual(summary["observed_regain_fraction_all_chains"], 1 / 3)
        self.assertTrue(records[0]["outcome_conflict"])
        self.assertEqual(records[1]["quality"]["fully_extrapolated"], {"unknown": 1})

    def test_chain_missing_identity_is_not_combined_into_fake_chain(self):
        summary, _ = summarize_pressing_chains([chain("a", period=""), chain("b", team_id="")], "1")
        self.assertEqual(summary["chains"], 0)
        self.assertEqual(summary["members_missing_chain_identity"], 2)
        self.assertIsNone(summary["regain_rate_known"])

    def test_supplier_components_and_option_peak_values_do_not_become_decision_scores(self):
        pp = possession(pass_epv_total="0.03", pass_epv_delta_for="0.05", pass_epv_delta_against="0.02",
                        possession_epv_total="0.1", possession_epv_delta_for="0.2", possession_epv_delta_against="0.1")
        cases, _ = build_pass_contexts([pp, option(xthreat="999", xpass_completion="999", peak_passing_option_frame="30")], "1")
        summary = summarize_passes(cases)
        epv = summary["supplier_epv"]
        self.assertEqual(epv["pass_epv_delta_for"]["mean"], 0.05)
        self.assertEqual(epv["pass_epv_delta_against"]["mean"], 0.02)
        self.assertNotIn("xthreat", cases[0])
        self.assertNotIn("regret", json.dumps(summary))

    def write_match(self, root, mid, date):
        folder = root / str(mid)
        folder.mkdir()
        (folder / f"{mid}_match.json").write_text(json.dumps({
            "id": mid, "date_time": date, "pitch_length": 105, "pitch_width": 68,
        }))
        rows = [dict(possession(pass_outcome="successful"), match_id=str(mid)),
                dict(option(), match_id=str(mid))]
        fields = sorted({key for row in rows for key in row})
        with (folder / f"{mid}_dynamic_events.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_run_uses_complete_chronological_matches_with_input_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "matches"
            root.mkdir()
            for mid, date in [(5, "2025-01-01T12:00:00Z"), (2, "2025-03-01T12:00:00Z"),
                              (9, "2025-02-01T12:00:00Z")]:
                self.write_match(root, mid, date)
            report = run(root, Path(tmp) / "output")
            self.assertEqual(report["matches"], 3)
            self.assertEqual(report["passing"]["pass_attempts"], 3)
            self.assertEqual(report["chronological_split"]["earlier"]["match_ids"], ["5", "9"])
            self.assertEqual(report["chronological_split"]["later"]["match_ids"], ["2"])
            self.assertFalse(report["chronological_split"]["model_trained"])
            provenance = json.loads((Path(tmp) / "output/provenance.json").read_text())
            self.assertEqual(len(provenance["files"]), 6)
            self.assertTrue(all(len(r["sha256"]) == 64 for r in provenance["files"]))
            self.assertEqual(report["run_id"], run(root, Path(tmp) / "repeat")["run_id"])
            json.dumps(report, allow_nan=False)

    def test_same_kickoff_matches_are_not_split_across_cohorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "matches"
            root.mkdir()
            for mid in range(1, 4):
                self.write_match(root, mid, "2025-01-01T12:00:00Z")
            report = run(root, Path(tmp) / "output")
            self.assertEqual(report["chronological_split"]["later"]["match_ids"], [])
            self.assertEqual(report["chronological_split"]["status"], "insufficient_distinct_match_dates")

    def test_no_source_matches_or_mismatched_metadata_fail_without_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                run(root, root / "output")
            self.write_match(root, 1, "2025-01-01T12:00:00Z")
            (root / "1/1_match.json").write_text(json.dumps({"id": 2}))
            with self.assertRaises(ValueError):
                run(root, root / "output")
            self.assertFalse((root / "output/report.json").exists())
