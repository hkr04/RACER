"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
from fastchat.utils import str_to_torch_dtype

from evaluation.eval import run_eval, reorg_answer_file

from racer.model.utils import *
from racer.model.racer_model import RacerModel
from racer.model.kv_cache import initialize_past_key_values

def baseline_forward(inputs, model, tokenizer, max_new_tokens, max_tokens, temperature=0.0, top_p=0.0, extra_args={}):
    input_ids = inputs.input_ids
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data_list = model.past_key_values_data_list
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data_list,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data_list = past_key_values_data_list
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]

    model.set_tree_mask(None)
    position_ids = torch.arange(
        input_ids.shape[1], dtype=torch.long, device=input_ids.device
    )[None, :]
    outputs = model.base_model(
        input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        position_ids=position_ids,
        **extra_args
    )

    for idx in range(max_new_tokens):
        if top_p > 0:
            assert top_p < 1, "top_p should between 0.0 and 1"
            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            input_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            input_id = input_id.view(input_id.shape[0], 1)
        else:
            input_id = outputs.logits[:, -1:].argmax(dim=-1)
        position_ids = torch.tensor([[input_ids.size(-1)]], device=input_ids.device)
        outputs = model.base_model(input_id, use_cache=True, past_key_values=past_key_values, position_ids=position_ids)
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        if model.tokenizer.eos_token_id in input_ids[0, input_len:]:
            break
        if input_ids.size(-1) + 1 > max_tokens:
            break

    accept_length_list = [1] * (idx + 1)
    return input_ids, idx+1, idx+1, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
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
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="The maximum number of total tokens. Default to config.max_position_embeddings"
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
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="The threshold for nucleus sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )

    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")
    
    model = RacerModel.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    if args.temperature == 0:
        args.top_p = 0
        
    if args.max_tokens is None:
        args.max_tokens = model.base_model.config.max_position_embeddings if hasattr(model.base_model.config, "max_position_embeddings") else 2048

    run_eval(
        model=model,
        tokenizer=model.get_tokenizer(),
        forward_func=baseline_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        max_tokens=args.max_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    reorg_answer_file(answer_file)