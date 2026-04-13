"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse

from evaluation.eval import run_eval, reorg_answer_file

from fastchat.utils import str_to_torch_dtype

from racer.model.utils import *
from racer.model.racer_model import RacerModel
from racer.model.kv_cache import initialize_past_key_values


@torch.no_grad()
def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10):
    input_length = input_ids.size(1)

    # Ensure max_ngram_size and num_pred_tokens are valid
    if max_ngram_size <= 0 or num_pred_tokens <= 0 or max_ngram_size > input_length:
        raise ValueError("Invalid max_ngram_size or num_pred_tokens")

    for ngram_size in range(max_ngram_size, 0, -1):
        # Extract the last n tokens as our search ngram
        ngram = input_ids[0, -ngram_size:].tolist()

        # Create sliding windows of size ngram_size
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

        # Convert ngram to a tensor for comparison
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)

        # Find where the windows match the ngram
        matches = (windows == ngram_tensor).all(dim=2)

        # Get the indices of matches
        match_indices = matches.nonzero(as_tuple=True)[1]

        # Iterate through match indices to find a valid continuation
        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            # Ensure we don't go beyond the length of input_ids and avoid self-match
            if end_idx <= input_length and start_idx < input_length - ngram_size:
                return input_ids[0, start_idx:end_idx]

    # If no match is found, return an empty tensor
    return torch.tensor([], dtype=torch.long, device=input_ids.device)

def pld_forward(inputs, model, tokenizer, max_new_tokens, max_tokens, temperature, top_p, extra_args={}):
    input_ids = inputs.input_ids
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    accept_length_list = []

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
    cur_length = input_len
    model.set_tree_mask(None)
    logits = initialize_logits(
        input_ids, model, past_key_values
    )
    new_token = 0

    for idx in range(max_new_tokens): 
        if top_p > 0:
            assert top_p < 1, "top_p should between 0.0 and 1"
            next_token_logits = logits[:, -1, :]
            next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            next_token = next_token.view(next_token.shape[0], 1)
        else:
            next_token = logits[:, -1:].argmax(dim=-1)
        
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Degraded tree -> chain
        candidates = torch.cat([next_token, find_candidate_pred_tokens(input_ids).unsqueeze(0)], dim=-1)
        chain_position_ids = torch.arange(0, candidates.shape[1]).unsqueeze(0).to(input_ids.device)
        
        # Temporarily remove next-token
        input_ids = input_ids[:, :-1]
        
        # Compute new position IDs by adding the draft position IDs to the length of the input sequence.
        position_ids = chain_position_ids + input_ids.shape[1]

        outputs, logits = model(
            candidates,
            output_orig=True,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        
        # Only one candidate, best_candidate must be 0
        best_candidate, accept_length = evaluate_posterior(
            logits, candidates, temperature, top_p
        )
        # Just for compatibility
        retrieve_indices = chain_position_ids
        input_ids, logits, new_token, accept_length = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            retrieve_indices,
            outputs,
            logits,
            new_token,
            past_key_values_data_list,
            current_length_data,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        if tokenizer.eos_token_id == input_ids[0, -1]:
            break
        if new_token > max_new_tokens:
            break
        if input_ids.size(-1) + 10 > max_tokens:
            break
    return input_ids, new_token, idx+1, accept_length_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
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
        help="The temperature for sampling.",
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
        import time
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}_{timestamp}.jsonl"

    print(f"Output to {answer_file}")
    
    model = RacerModel.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()
    
    if args.temperature == 0:
        args.top_p = 0
        
    if args.max_tokens is None:
        args.max_tokens = model.base_model.config.max_position_embeddings

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=pld_forward,
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