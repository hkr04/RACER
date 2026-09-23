"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import torch
import argparse
from fastchat.utils import str_to_torch_dtype

from evaluation.eval import (
    reorg_answer_file,
    run_eval,
    str2bool,
)

from evaluation.model.eagle3.eagle3_model import Eagle3Model
from evaluation.model.eagle3.utils import *

from racer.model.chat_template import VICUNA_CHAT_TEMPLATE

def ea_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0, top_p=0.0, extra_args={}):
    input_ids = inputs.input_ids
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    input_ids, new_token, step, accept_length_list = model.eagle_generate(
        torch.as_tensor(input_ids).cuda(),
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        log=True
    )

    return input_ids, new_token, step, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eagle-model-path",
        type=str,
        default="down_checkpoints/LC70B",
        help="Path to the weights (local folder or Hugging Face repo ID)",
    )
    parser.add_argument("--base-model-path", type=str, default="")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="ess-vicuna-70b-fp16")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="spec_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--enable-thinking",
        type=str2bool,
        default=None,
        help=(
            "Put enable_thinking in extra_args for the chat template. "
            "Use False to disable thinking on models such as Qwen3. "
            "Omit it to keep the template default."
        ),
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=60,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="The threshold for nucleus sampling.",
    )
    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )

    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    model = Eagle3Model.from_pretrained(
        base_model_path=args.base_model_path,
        eagle_model_path=args.eagle_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()
    
    if "vicuna" in args.base_model_path.lower():
        tokenizer.chat_template = VICUNA_CHAT_TEMPLATE

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=ea_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        bench_name=args.bench_name,
        extra_args=(
            {} if args.enable_thinking is None
            else {"enable_thinking": args.enable_thinking}
        ),
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    reorg_answer_file(answer_file)
