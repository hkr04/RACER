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
from racer.automaton import Automaton


def racer_forward(inputs, model, tokenizer, max_new_tokens, max_tokens, temperature, top_p, ac, max_num_draft, ngram, max_breadth, extra_args={}):
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
        input_ids, model, past_key_values, extra_args
    )
    new_token = 0
    
    if max_breadth:
        adj_topk = torch.topk(logits, k=max_breadth, dim=-1)[1][0]
        ac.update(input_ids[0].tolist(), adj_topk.tolist())
    
    if ngram:
        for i in range(input_len):
            start = i
            end = i + ngram
            if end > input_ids.size(1):
                break
            pattern = input_ids[0, start:end].tolist()
            ac.insert(pattern)
        ac.build()
    
    for idx in range(max_new_tokens): 
        if max_tokens - input_ids.size(-1) < max_num_draft:
            max_num_draft = max_tokens - input_ids.size(-1)
        # Automaton will receive the next token predicted by the logit in this function
        candidates, tree_candidates, tree_attn_mask, tree_position_ids, retrieve_indices = generate_draft_tree(
            logits=logits,
            ac=ac,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            top_p=top_p,
            temperature=temperature,
            max_num_draft=max_num_draft,
            device=model.base_model.device
        )
        tree_candidates = tree_candidates[None, :]
        tree_attn_mask = tree_attn_mask[None, None, :]
        tree_position_ids = tree_position_ids[None, :]
        model.set_tree_mask(tree_attn_mask)
        logits, outputs, tree_logits = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            tree_position_ids,
            input_ids,
            retrieve_indices
        )
        
        best_candidate, accept_length = evaluate_posterior(
            logits, candidates, temperature, top_p
        )
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
        
        if max_breadth:
            adj_topk = torch.topk(tree_logits, k=max_breadth, dim=-1)[1][0]
            ac.update(tree_candidates[0].tolist(), adj_topk.tolist())
                
        if ngram:
            for i in range(1 + accept_length):
                start = -ngram - i
                end = -i if i != 0 else None
                if abs(start) > input_ids.size(1):
                    break
                pattern = input_ids[0, start:end].tolist()
                ac.insert(pattern)
            if accept_length > 0:
                ac.trans_tokens(input_ids[0, -accept_length:].tolist())
        
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        if tokenizer.eos_token_id == input_ids[0, -1]:
            break
        if new_token > max_new_tokens:
            break
        if input_ids.size(-1) + 2 > max_tokens:
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
        "--max-nodes",
        type=int,
        default=10000,
        help="The maximum number of nodes in the automaton.",
    )
    parser.add_argument(
        "--max-num-draft",
        type=int,
        default=64,
        help="The maximum number of draft tokens",
    )
    parser.add_argument(
        "--ngram",
        type=int,
        default=10,
        help="The ngram for retrieval.",
    )
    parser.add_argument(
        "--max-breadth",
        type=int,
        default=8,
        help="The maximum breadth for logits draft tree."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )

    args = parser.parse_args()

    if args.temperature == 0:
        args.top_p = 0

    args.model_id = args.model_id \
    + "-temperature-" + str(args.temperature) \
    + "-top_p-" + str(args.top_p) \
    + "-draft-" + str(args.max_num_draft) \
    + "-n-" + str(args.ngram) \
    + "-node-" + str(args.max_nodes) \
    + "-breadth-" + str(args.max_breadth)

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
    
    print("Initializing automaton ...")
    ac = Automaton(args.max_nodes)
    if args.max_breadth > 0:
        ac.init_logits(tokenizer.vocab_size, args.max_breadth)
        
    if args.max_tokens is None:
        args.max_tokens = model.base_model.config.max_position_embeddings if hasattr(model.base_model.config, "max_position_embeddings") else 2048

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=racer_forward,
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
        ac=ac,
        max_num_draft=args.max_num_draft,
        max_breadth=args.max_breadth,
        ngram=args.ngram
    )

    reorg_answer_file(answer_file)