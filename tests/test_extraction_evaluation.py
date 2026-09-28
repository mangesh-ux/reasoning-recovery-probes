from __future__ import annotations

import unittest

from reasoning_recovery.evaluation import (
    EvaluationStatus,
    classify_transition,
    evaluate_extraction,
)
from reasoning_recovery.extraction import ExtractionStatus, extract_last_balanced_boxed


class FakeBackend:
    name = "fake-math"
    version = "test"

    def parse(self, text: str) -> list[str]:
        cleaned = text.replace("\\boxed{", "").replace("}", "").strip()
        return [cleaned] if cleaned else []

    def verify(self, reference: list[str], candidate: list[str]) -> bool:
        return reference == candidate


class ExtractionEvaluationTests(unittest.TestCase):
    def test_last_balanced_box_is_retained_with_nested_braces(self) -> None:
        extraction = extract_last_balanced_boxed(
            r"first \boxed{1}; final \boxed{\frac{2}{3}}"
        )
        self.assertEqual(extraction.status, ExtractionStatus.EXTRACTED)
        self.assertEqual(extraction.extracted_text, r"\boxed{\frac{2}{3}}")

    def test_missing_and_malformed_boxes_are_not_guessed(self) -> None:
        missing = extract_last_balanced_boxed("answer is probably four")
        malformed = extract_last_balanced_boxed(r"\boxed{4")
        self.assertEqual(missing.status, ExtractionStatus.MISSING_BOXED_ANSWER)
        self.assertEqual(malformed.status, ExtractionStatus.MALFORMED_BOXED_ANSWER)
        result = evaluate_extraction("4", missing, backend=FakeBackend())
        self.assertEqual(result.status, EvaluationStatus.NON_EVALUABLE)

    def test_later_balanced_box_wins_over_an_earlier_malformed_marker(self) -> None:
        extraction = extract_last_balanced_boxed(
            r"broken \boxed{unfinished answer; later \boxed{4}"
        )
        self.assertEqual(extraction.status, ExtractionStatus.EXTRACTED)
        self.assertEqual(extraction.extracted_text, r"\boxed{4}")

    def test_later_malformed_marker_does_not_discard_last_balanced_box(self) -> None:
        extraction = extract_last_balanced_boxed(r"answer \boxed{4}; typo \boxed{")
        self.assertEqual(extraction.status, ExtractionStatus.EXTRACTED)
        self.assertEqual(extraction.extracted_text, r"\boxed{4}")

    def test_evaluation_and_transition_are_explicit(self) -> None:
        final = evaluate_extraction("4", extract_last_balanced_boxed(r"\boxed{4}"), backend=FakeBackend())
        forced = evaluate_extraction("4", extract_last_balanced_boxed(r"\boxed{5}"), backend=FakeBackend())
        self.assertEqual(final.status, EvaluationStatus.CORRECT)
        self.assertEqual(forced.status, EvaluationStatus.INCORRECT)
        self.assertEqual(classify_transition(forced, final).value, "W_TO_C")


if __name__ == "__main__":
    unittest.main()
