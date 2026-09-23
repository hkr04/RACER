"""Extract \\boxed{} answers and score math-benchmark outputs."""

import json
import math
import re
from decimal import Decimal, InvalidOperation


_BOXED_RE = re.compile(r"\\boxed\s*\{")


def extract_boxed(text):
    """Return the contents of the last \\boxed{...}, or None if it is missing."""
    if not text:
        return None
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1]


def normalize_answer(answer):
    """Strip currency, commas, and whitespace, then parse a number when possible."""
    if answer is None:
        return None
    text = str(answer).strip()
    text = text.replace(",", "").replace("$", "").replace(" ", "")
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = text.strip()
    if text.endswith("."):
        text = text[:-1]
    try:
        return Decimal(text)
    except InvalidOperation:
        return text


def answers_match(prediction, reference):
    return normalize_answer(prediction) == normalize_answer(reference)


def _choice_texts(row):
    """First-turn text of every sampled choice. Missing rows count as one empty sample."""
    if not row or not row.get("choices"):
        return [""]
    texts = []
    for choice in row["choices"]:
        turns = choice.get("turns") or []
        texts.append(turns[0] if turns else "")
    return texts or [""]


def score_math(questions, answer_file):
    """Score every sampled choice against question references.

    Questions without a reference are skipped. A sample without a boxed answer
    counts as incorrect and as boxed_missing.

    With one sample per question this is accuracy. With k samples, pass@k is
    the fraction of questions with at least one correct sample, and mean@k is
    the average of the k pass@1 scores. std is the population standard
    deviation of those k pass@1 scores.
    """
    answers = {}
    with open(answer_file, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            answers[row["question_id"]] = row

    total = 0
    boxed_missing = 0
    pass_correct = 0
    per_question_hits = []
    sample_counts = set()
    for question in questions:
        if "reference" not in question:
            continue
        total += 1
        texts = _choice_texts(answers.get(question["question_id"]))
        sample_counts.add(len(texts))
        hits = []
        for text in texts:
            prediction = extract_boxed(text)
            if prediction is None:
                boxed_missing += 1
                hits.append(False)
                continue
            hits.append(answers_match(prediction, question["reference"]))
        if any(hits):
            pass_correct += 1
        per_question_hits.append(hits)

    if total == 0:
        return {"correct": 0, "total": 0, "boxed_missing": 0}

    k = sample_counts.pop() if len(sample_counts) == 1 else None
    mean_value = sum(sum(hits) / len(hits) for hits in per_question_hits) / total
    pass_accuracy = pass_correct / total * 100
    mean_accuracy = mean_value * 100
    if k == 1:
        print(
            f"Math: {pass_correct}/{total} = {pass_accuracy:.2f}% "
            f"(boxed_missing={boxed_missing})"
        )
        return {
            "correct": pass_correct,
            "total": total,
            "boxed_missing": boxed_missing,
        }

    if k is None:
        std_value = 0.0
    else:
        pass_at_1 = [
            sum(hits[i] for hits in per_question_hits) / total
            for i in range(k)
        ]
        pass_at_1_mean = sum(pass_at_1) / k
        std_value = math.sqrt(
            sum((value - pass_at_1_mean) ** 2 for value in pass_at_1) / k
        )
    std_accuracy = std_value * 100
    k_label = str(k) if k is not None else "k"
    print(
        f"Math: pass@{k_label} = {pass_correct}/{total} = {pass_accuracy:.2f}%, "
        f"mean@{k_label} = {mean_accuracy:.2f}% ± {std_accuracy:.2f}% "
        f"(boxed_missing={boxed_missing})"
    )
    return {
        "correct": pass_correct,
        "total": total,
        "boxed_missing": boxed_missing,
        "k": k,
        "pass_at_k": pass_correct / total,
        "mean_at_k": mean_value,
        "std_at_k": std_value,
    }
