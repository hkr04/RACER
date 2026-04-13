"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
# adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import json
import os
import time
import torch
import numpy as np
import shortuuid
import traceback

from fastchat.llm_judge.common import load_questions
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor
from qwen_vl_utils import process_vision_info

def run_eval(
        model,
        tokenizer,
        forward_func,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_tokens,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        **kwargs,
):
    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers
        
    Type = None
    
    try:
        Type = AutoConfig.from_pretrained(model.base_model_name_or_path, trust_remote_code=True).architectures[0]
    except Exception as e:
        pass
                
    if Type in ["LlavaForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration", "Qwen3VLForConditionalGeneration"]:
        processor = AutoProcessor.from_pretrained(model.base_model_name_or_path)
    else:
        processor = None

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model,
                tokenizer,
                processor,
                forward_func,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_tokens,
                num_choices,
                **kwargs,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        processor,
        forward_func,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        num_choices,
        **kwargs,
):

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)
        messages = []
        if "vicuna" in model_id:
            messages.append({
                "role": "system",
                "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            })
        else: # default
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant."
            })
        turns = []
        steps = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            messages.append({
                "role": "user",
                "content": qs
            })
            extra_args = {}
            if processor:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.base_model.device)
                input_ids = inputs.input_ids
                if "pixel_values" in inputs:
                    extra_args["pixel_values"] = inputs.pixel_values
                if "image_grid_thw" in inputs:
                    extra_args["image_grid_thw"] = inputs.image_grid_thw
                if "pixel_values_videos" in inputs:
                    extra_args["pixel_values_videos"] = inputs.pixel_values_videos
                if "video_grid_thw" in inputs:
                    extra_args["video_grid_thw"] = inputs.video_grid_thw
            else:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.base_model.device)
                input_ids = inputs.input_ids
            try:
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, step, accept_length_tree = forward_func(
                    inputs,
                    model,
                    tokenizer,
                    max_new_tokens,
                    extra_args=extra_args,
                    **kwargs,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]):]
                # be consistent with the template's eos_token_id
                if tokenizer.eos_token_id is not None:
                    eos_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id == tokenizer.eos_token_id
                    ]
                    if len(eos_index) > 0:
                        output_ids = output_ids[: eos_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

            except RuntimeError as e:
                print("ERROR question ID: ", question["question_id"])
                print("Exception: ", e)
                traceback.print_exc()
                output = "ERROR"

            turns.append(output)
            steps.append(int(step))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            messages.append({
                "role": "assistant",
                "content": output
            })
    print('Warmup done')

    accept_lengths_tree = []
    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            cur_accept_lengths_tree = []
            torch.manual_seed(i)
            messages = []
            if "vicuna" in model_id:
                messages.append({
                    "role": "system",
                    "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
                })
            else: # default
                messages.append({
                    "role": "system",
                    "content": "You are a helpful assistant."
                })
            turns = []
            steps = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append({
                    "role": "user",
                    "content": qs
                })
                extra_args = {}
                if processor:
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(model.base_model.device)
                    input_ids = inputs.input_ids
                    if "pixel_values" in inputs:
                        extra_args["pixel_values"] = inputs.pixel_values
                    if "image_grid_thw" in inputs:
                        extra_args["image_grid_thw"] = inputs.image_grid_thw
                    if "pixel_values_videos" in inputs:
                        extra_args["pixel_values_videos"] = inputs.pixel_values_videos
                    if "video_grid_thw" in inputs:
                        extra_args["video_grid_thw"] = inputs.video_grid_thw
                else:
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.base_model.device)
                    input_ids = inputs.input_ids
                try:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    output_ids, new_token, step, accept_length_tree = forward_func(
                        inputs,
                        model,
                        tokenizer,
                        max_new_tokens,
                        extra_args=extra_args,
                        **kwargs,
                    )
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    accept_lengths_tree.extend(accept_length_tree)
                    output_ids = output_ids[0][len(input_ids[0]):]

                    # be consistent with the template's eos_token_id
                    if tokenizer.eos_token_id is not None:
                        eos_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id == tokenizer.eos_token_id
                        ]
                        if len(eos_index) > 0:
                            output_ids = output_ids[: eos_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    print("Exception: ", e)
                    traceback.print_exc()
                    output = "ERROR"

                turns.append(output)
                steps.append(int(step))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                cur_accept_lengths_tree.extend(accept_length_tree)
                messages.append({
                    "role": "assistant",
                    "content": output
                })
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                            "accept_lengths": cur_accept_lengths_tree})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")
    print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])

