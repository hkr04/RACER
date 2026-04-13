# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/cli.py
# Adapted from: https://github.com/FasterDecoding/Medusa/blob/main/medusa/inference/cli.py
"""
Chat with a model with command line interface.

Usage:
python3 -m racer.inference.cli --model-path <model_name_or_path>
Other commands:
- Type "!!exit" or an empty line to exit.
- Type "!!reset" to start a new conversation.
- Type "!!remove" to remove the last prompt.
- Type "!!regen" to regenerate the last message.
- Type "!!save <filename>" to save the conversation history to a json file.
- Type "!!load <filename>" to load a conversation history from a json file.
- Use <image><image_path></image> to insert an image in the prompt.
"""
import argparse
import os
import re
import sys
import torch
from transformers import AutoConfig
from fastchat.serve.cli import SimpleChatIO, RichChatIO, ProgrammaticChatIO
from fastchat.model.model_adapter import get_conversation_template
from fastchat.conversation import get_conv_template
from fastchat.utils import str_to_torch_dtype
import json
from racer.model.racer_model import RacerModel
from racer.automaton import Automaton

IMAGE_START = "<image>"
IMAGE_END = "</image>"

def main(args):
    if args.style == "simple":
        chatio = SimpleChatIO(args.multiline)
    elif args.style == "rich":
        chatio = RichChatIO(args.multiline, args.mouse)
    elif args.style == "programmatic":
        chatio = ProgrammaticChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    try:
        model = RacerModel.from_pretrained(
            args.model_path,
            torch_dtype=str_to_torch_dtype(args.dtype),
            low_cpu_mem_usage=True,
            device_map="auto",
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit
        )
        tokenizer = model.get_tokenizer()
        if not args.baseline:
            ac = Automaton(max_nodes=args.max_nodes)
            if args.max_breadth > 0:
                ac.init_logits(tokenizer.vocab_size, args.max_breadth)
                
        arch = model.config.architectures[0]

        if "ConditionalGeneration" in arch:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(args.model_path)
        else:
            processor = None

        messages = []
        
        if args.system_prompt:
            messages.append({
                "role": "system",
                "content": args.system_prompt
            })
            
        offset = len(messages)

        def reload_messages(messages):
            """
            Reprints the conversation from the start.
            """
            for message in messages[offset:]:
                chatio.prompt_for_output(message["role"])
                chatio.print_output(message["content"])

        while True:
            try:
                inp = chatio.prompt_for_input("user")
            except EOFError:
                inp = ""

            if inp == "!!exit" or not inp:
                print("exit...")
                break
            elif inp == "!!reset":
                print("resetting...")
                messages = []
                if args.system_prompt:
                    messages.append({
                        "role": "system",
                        "content": args.system_prompt
                    })
                continue
            elif inp == "!!remove":
                print("removing last message...")
                if len(messages) > conv.offset:
                    # Assistant / User
                    if messages[-1]["role"] in ["user", "assistant"]:
                        messages.pop()
                    reload_messages(messages)
                else:
                    print("No messages to remove.")
                continue
            elif inp == "!!regen":
                print("regenerating last message...")
                if len(messages) > offset:
                    # Assistant
                    if messages[-1]["role"] == "assistant":
                        messages.pop()
                    # User
                    if messages[-1]["role"] == "user":
                        reload_messages(messages)
                        # Set inp to previous message
                        inp = messages.pop()["content"]
                    else:
                        # Shouldn't happen in normal circumstances
                        print("No user message to regenerate from.")
                        continue
                else:
                    print("No messages to regenerate.")
                    continue
            elif inp.startswith("!!save"):
                args = inp.split(" ", 1)

                if len(args) != 2:
                    print("usage: !!save <filename>")
                    continue
                else:
                    filename = args[1]

                # Add .json if extension not present
                if not "." in filename:
                    filename += ".json"

                print("saving...", filename)
                with open(filename, "w") as outfile:
                    json.dump(messages, outfile)
                continue
            elif inp.startswith("!!load"):
                args = inp.split(" ", 1)

                if len(args) != 2:
                    print("usage: !!load <filename>")
                    continue
                else:
                    filename = args[1]

                # Check if file exists and add .json if needed
                if not os.path.exists(filename):
                    if (not filename.endswith(".json")) and os.path.exists(
                        filename + ".json"
                    ):
                        filename += ".json"
                    else:
                        print("file not found:", filename)
                        continue

                print("loading...", filename)
                with open(filename, "r") as infile:
                    messages = json.load(infile)

                reload_messages(messages)
                continue
            
            messages.append({
                "role": "user",
                "content": []
            })
            
            while IMAGE_START in inp and IMAGE_END in inp:
                start_idx = inp.index(IMAGE_START)
                end_idx = inp.index(IMAGE_END, start_idx) + len(IMAGE_END)
                image_path = inp[start_idx + len(IMAGE_START): end_idx - len(IMAGE_END)].strip()
                
                if start_idx != 0:
                    messages[-1]["content"].append({
                        "type": "text",
                        "text": inp[:start_idx]
                    })
                
                if os.path.exists(image_path): # Local file path
                    messages[-1]["content"].append({
                        "type": "image",
                        "image": "file://" + image_path
                    })
                else: # Image URL
                    messages[-1]["content"].append({
                        "type": "text",
                        "image": image_path
                    })
                
                inp = inp[end_idx:]
                
            
            if inp:
                messages[-1]["content"].append({
                    "type": "text",
                    "text": inp
                })
                
            if processor is None and len(messages[-1]["content"]) == 1 and messages[-1]["content"][0]["type"] == "text": # Fallback to text only
                messages[-1]["content"] = messages[-1]["content"][0]["text"]

            try:
                chatio.prompt_for_output("assistant")
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
                    input_ids = tokenizer(text, return_tensors="pt", padding=False).to(model.base_model.device).input_ids
                outputs = chatio.stream_output(
                    model.racer_generate(
                        input_ids,
                        ac,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_steps=args.max_steps,
                        max_num_draft=args.max_num_draft,
                        max_breadth=args.max_breadth,
                        ngram=args.ngram,
                        show_accepted=True,
                        debug_logits_path=args.debug_logits_path,
                        extra_args=extra_args
                    ) if not args.baseline else \
                    model.baseline_generate(
                        input_ids,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_steps=args.max_steps,
                        debug_logits_path=args.debug_logits_path,
                        extra_args=extra_args
                    )
                )
                # Clean color codes
                outputs = re.sub(r"\x1b\[[0-9;]*m", "", outputs)
                messages.append({
                    "role": "assistant",
                    "content": outputs
                })

            except KeyboardInterrupt:
                print("stopped generation.")
                # Remove last user message, so there isn't a double up
                if messages[-1]["role"] == "user":
                    messages.pop()

                reload_messages(messages)

    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="qwen/qwen3-1.7b", help="LLM name or path.")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.", help="System prompt.")
    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--baseline", action="store_true", help="Use standard autoregressive generation.")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    parser.add_argument(
        "--debug-logits-path",
        type=str,
        default=None,
        help="Path to save debug logits."
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=1000,
        help="Maximum number of nodes to cache."
    )
    parser.add_argument(
        "--max-num-draft",
        type=int,
        default=64,
        help="Maximum number of draft tokens to reuse in the context.",
    )
    parser.add_argument(
        "--ngram",
        type=int,
        default=10
    )
    parser.add_argument(
        "--max-breadth",
        type=int,
        default=8,
        help="The maximum breadth for logits draft tree."
    )
    
    args = parser.parse_args()
    
    if args.temperature == 0:
        args.top_p = 0
        
    main(args)
