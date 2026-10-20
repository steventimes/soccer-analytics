import unittest
from app.data_service.tactical_evidence import validate_card


class TestTacticalEvidence(unittest.TestCase):
    def test_metadata_and_show_notes_cannot_become_coach_statements(self):
        for level in ["metadata_only", "blocked", "show_notes"]:
            with self.assertRaises(ValueError):
                validate_card(
                    {"content_level": level}, {"claim_type": "coach_self_report"}, ""
                )
        validate_card(
            {"content_level": "show_notes"},
            {"claim_type": "topic_only", "anchor": "rest defence"},
            "rest defence is a programme topic",
        )

    def test_statement_requires_an_anchor_in_the_inspected_text(self):
        source = {"content_level": "automatic_captions"}
        card = {
            "claim_type": "coach_self_report",
            "anchor": "we wanted Gomez on the ball",
        }
        validate_card(
            source, card, "So we wanted Gomez on the ball and waited for the next pass"
        )
        validate_card(
            source, card, "So we wanted Gomez\non the ball and waited for the next pass"
        )
        with self.assertRaises(ValueError):
            validate_card(source, card, "No such statement")
