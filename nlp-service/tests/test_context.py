import unittest

from app.context.scoring import score_context_requirement
from app.context.selector import select_context_messages


class CanonicalContextExamplesTest(unittest.TestCase):
    def test_internship_follow_up_uses_the_relevant_recent_messages(self):
        history = [
            "we discussed the internship yesterday",
            "The recruiter replied.",
            "They rejected it.",
        ]

        result = score_context_requirement("That's disappointing.", history)
        selected = select_context_messages(
            result.considered_history, result.similarities, result.level
        )

        self.assertEqual(result.level, "medium")
        self.assertEqual(
            selected,
            ["The recruiter replied.", "They rejected it."],
        )

    def test_thanks_after_an_unrelated_exchange_needs_no_context(self):
        history = ["The weather is nice today.", "Yes, it is sunny."]

        result = score_context_requirement("Thanks!", history)
        selected = select_context_messages(
            result.considered_history, result.similarities, result.level
        )

        self.assertEqual(result.level, "low")
        self.assertEqual(selected, [])

    def test_correction_follow_up_is_context_dependent(self):
        history = ["I finally got the results.", "That's great!"]

        result = score_context_requirement(
            "No, I meant the other results.", history
        )
        selected = select_context_messages(
            result.considered_history, result.similarities, result.level
        )

        self.assertEqual(result.level, "medium")
        self.assertEqual(selected, history)


if __name__ == "__main__":
    unittest.main()
