import json
import os
import tempfile
import unittest

from evaluation.eval import model_extra_args, render_chat
from evaluation.math_score import (
    answers_match,
    extract_boxed,
    score_math,
)


class ExtractBoxedTest(unittest.TestCase):
    def test_last_box(self):
        text = r"first \boxed{1} then the answer is \boxed{18}"
        self.assertEqual(extract_boxed(text), "18")

    def test_nested_braces(self):
        text = r"therefore \boxed{\dfrac{1}{2}}"
        self.assertEqual(extract_boxed(text), r"\dfrac{1}{2}")

    def test_thousands_separator_matches_reference(self):
        self.assertTrue(answers_match(r"5,600", "5600"))
        self.assertTrue(answers_match(extract_boxed(r"\boxed{$5,600}"), "5600"))

    def test_missing_box(self):
        self.assertIsNone(extract_boxed("the answer is 18"))
        self.assertIsNone(extract_boxed(""))
        self.assertIsNone(extract_boxed(None))


class ScoreMathTest(unittest.TestCase):
    def test_scores_first_choice_and_counts_missing_box(self):
        questions = [
            {"question_id": 1, "reference": "18"},
            {"question_id": 2, "reference": "5600"},
            {"question_id": 3, "reference": "3"},
        ]
        rows = [
            {
                "question_id": 1,
                "choices": [{"turns": [r"work \boxed{1} final \boxed{18}"]}],
            },
            {
                "question_id": 2,
                "choices": [{"turns": [r"\boxed{5,600}"]}],
            },
            {
                "question_id": 3,
                "choices": [{"turns": ["no box here"]}],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            answer_file = os.path.join(tmp, "answers.jsonl")
            with open(answer_file, "w", encoding="utf-8") as fout:
                for row in rows:
                    fout.write(json.dumps(row) + "\n")
            result = score_math(questions, answer_file)
        self.assertEqual(result, {"correct": 2, "total": 3, "boxed_missing": 1})

    def test_skips_questions_without_reference(self):
        questions = [{"question_id": 1, "turns": ["no gold answer"]}]
        with tempfile.TemporaryDirectory() as tmp:
            answer_file = os.path.join(tmp, "answers.jsonl")
            with open(answer_file, "w", encoding="utf-8") as fout:
                fout.write('{"question_id": 1, "choices": [{"turns": ["\\\\boxed{1}"]}]}\n')
            result = score_math(questions, answer_file)
        self.assertEqual(result, {"correct": 0, "total": 0, "boxed_missing": 0})

    def test_pass_and_mean_at_k(self):
        questions = [
            {"question_id": 1, "reference": "18"},
            {"question_id": 2, "reference": "3"},
            {"question_id": 3, "reference": "5"},
            {"question_id": 4, "reference": "7"},
        ]
        rows = [
            {
                "question_id": 1,
                "choices": [
                    {"turns": [r"\boxed{18}"]},
                    {"turns": [r"\boxed{1}"]},
                ],
            },
            {
                "question_id": 2,
                "choices": [
                    {"turns": [r"\boxed{0}"]},
                    {"turns": [r"\boxed{1}"]},
                ],
            },
            {
                "question_id": 3,
                "choices": [
                    {"turns": [r"\boxed{5}"]},
                    {"turns": [r"\boxed{5}"]},
                ],
            },
            {
                "question_id": 4,
                "choices": [
                    {"turns": ["no box"]},
                    {"turns": [r"\boxed{7}"]},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            answer_file = os.path.join(tmp, "answers.jsonl")
            with open(answer_file, "w", encoding="utf-8") as fout:
                for row in rows:
                    fout.write(json.dumps(row) + "\n")
            result = score_math(questions, answer_file)
        self.assertEqual(result["k"], 2)
        self.assertEqual(result["correct"], 3)
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["boxed_missing"], 1)
        self.assertAlmostEqual(result["pass_at_k"], 0.75)
        self.assertAlmostEqual(result["mean_at_k"], 0.5)
        # choice 0 pass@1 = 2/4, choice 1 pass@1 = 2/4
        self.assertAlmostEqual(result["std_at_k"], 0.0)

    def test_std_is_across_pass_at_1_runs(self):
        questions = [
            {"question_id": 1, "reference": "1"},
            {"question_id": 2, "reference": "2"},
            {"question_id": 3, "reference": "3"},
        ]
        rows = [
            {"question_id": 1, "choices": [{"turns": [r"\boxed{1}"]}, {"turns": [r"\boxed{0}"]}]},
            {"question_id": 2, "choices": [{"turns": [r"\boxed{2}"]}, {"turns": [r"\boxed{0}"]}]},
            {"question_id": 3, "choices": [{"turns": [r"\boxed{0}"]}, {"turns": [r"\boxed{3}"]}]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            answer_file = os.path.join(tmp, "answers.jsonl")
            with open(answer_file, "w", encoding="utf-8") as fout:
                for row in rows:
                    fout.write(json.dumps(row) + "\n")
            result = score_math(questions, answer_file)
        self.assertAlmostEqual(result["mean_at_k"], 0.5)
        self.assertAlmostEqual(result["std_at_k"], 1 / 6)


class FakeTemplate:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.kwargs = kwargs
        return ""


class EnableThinkingTest(unittest.TestCase):
    def test_enable_thinking_stays_in_extra_args_for_the_template_only(self):
        template = FakeTemplate()
        extra_args = {"enable_thinking": False, "pixel_values": "kept"}
        render_chat(template, [], extra_args)
        self.assertIs(template.kwargs["enable_thinking"], False)
        self.assertEqual(model_extra_args(extra_args), {"pixel_values": "kept"})
        self.assertIn("enable_thinking", extra_args)

        template = FakeTemplate()
        render_chat(template, [], {})
        self.assertNotIn("enable_thinking", template.kwargs)


if __name__ == "__main__":
    unittest.main()
