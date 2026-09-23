# Benchmarks

`--bench-name` accepts `spec_bench` | `human_eval` | `mgsm` | `gsm8k` | `math` | `aime` (default: `spec_bench`).

Questions live in `data/<bench-name>/question.jsonl`. `--question-end` limits how many questions are run. Answers are written to `data/<bench-name>/model_answer/`.

`--enable-thinking` is a generation option, like `--max-new-tokens` and `--temperature`. It is placed in `extra_args` and passed to the chat template, not into the model forward. Omit it to keep the template default. For thinking models such as Qwen3, pass `--enable-thinking False` to turn thinking off.

## General

`spec_bench` measures mean accepted tokens and decoding speed. Answers are not scored for correctness.

| Bench | Questions | Source |
| --- | --- | --- |
| `spec_bench` | 480 | [Spec-Bench](https://github.com/hemingkx/Spec-Bench): 80 multi-turn prompts from MT-Bench (writing, roleplay, reasoning, math, coding, extraction, stem, humanities; 10 each) and 400 single-turn prompts (translation, summarization, qa, math reasoning, rag; 80 each). |

## Code

`human_eval` measures mean accepted tokens and decoding speed on code prompts. Answers are not scored for correctness.

| Bench | Questions | Source |
| --- | --- | --- |
| `human_eval` | 164 | [HumanEval](https://github.com/openai/human-eval): Each prompt asks the model to implement one Python function. |

## Math benchmarks

`gsm8k`, `math`, `aime`, and `mgsm` ask the model to put the final answer in `\boxed{}`, then score the last boxed value against each question's `reference`.

With one sample this is accuracy. With `--num-choices k` greater than 1, the report is `pass@k`, plus `mean@k` and the standard deviation of the k per-sample `pass@1` scores.

`aime` is harder than the others. Raise `--max-new-tokens` above the default of 1024 so the boxed answer is not cut off.

| Bench | Questions | Source |
| --- | --- | --- |
| `gsm8k` | 250 | First 250 problems of the [GSM8K](https://github.com/openai/grade-school-math) test set ([openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), `main` / `test`). `reference` is the number after `####`. With `--enable-thinking False` and `--question-end 20`, this is the sanity check. |
| `mgsm` | 250 | Full Chinese test set of [MGSM](https://github.com/google-research/url-nlp/tree/main/mgsm) ([juletxara/mgsm](https://huggingface.co/datasets/juletxara/mgsm), `zh` / `test`). `reference` is `answer_number`. |
| `math` | 252 | Subset of [Hendrycks MATH](https://github.com/hendrycks/math) ([EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math)): 50 problems from each of Level 1-5, plus 2 unlabeled problems. `reference` is the final answer in the official solution. |
| `aime` | 250 | 250 problems from [AIME 1983-2024](https://huggingface.co/datasets/gneubig/aime-1983-2024) ([gneubig/aime-1983-2024](https://huggingface.co/datasets/gneubig/aime-1983-2024)). `reference` is the integer answer. |
