# Adapted from: https://github.com/SafeAILab/EAGLE/blob/main/eagle/application/webui.py
import os
import time
import html

import gradio as gr
import argparse
from racer.model.racer_model import RacerModel
from racer.automaton import Automaton
import torch
from fastchat.model import get_conversation_template
from fastchat.utils import str_to_torch_dtype
from fastchat.serve.cli import SimpleChatIO
import re


def truncate_list(lst, num):
    if num not in lst:
        return lst

    first_index = lst.index(num)

    return lst[:first_index + 1]

def find_list_markers(text):
    pattern = re.compile(r'(?m)(^\d+\.\s|\n)')
    matches = pattern.finditer(text)
    return [(match.start(), match.end()) for match in matches]


def checkin(pointer, start, marker):
    for b, e in marker:
        if b <= pointer < e:
            return True
        if b <= start < e:
            return True
    return False

def highlight_text(text, text_list, color="black"):

    pointer = 0
    result = ""
    markers = find_list_markers(text)


    for sub_text in text_list:

        start = text.find(sub_text, pointer)
        if start == -1:
            continue
        end = start + len(sub_text)


        plain_text = html.escape(text[pointer:start])

        if checkin(pointer, start, markers):
            result += plain_text
        else:
            result += f"<span style='color: {color};'>{plain_text}</span>"

        result += html.escape(sub_text)

        pointer = end

    if pointer < len(text):
        result += f"<span style='color: {color};'>{html.escape(text[pointer:])}</span>"

    return result


def warmup(model, tokenizer, ac):
    messages = [
        {"role": "user", "content": "hello"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt", padding=False).to(model.base_model.device).input_ids
    SimpleChatIO().stream_output(
        model.racer_generate(
            input_ids,
            ac,
            max_steps=128
        )
    )
        

def bot(history, temperature, top_p, use_racer, highlight_racer, session_state):
    if not history:
        return history, "0.00 tokens/s", "0.00", session_state
    pure_history = session_state.get("pure_history", [])

    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }]

    for query, response in pure_history:
        messages.append({
            "role": "user",
            "content": query
        })
        if response != None:
            messages.append({
                "role": "assistant",
                "content": response
            })

    prompt = model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    input_ids = model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    input_len = input_ids.shape[1]
    naive_text = []
    cur_len = input_len
    totaltime = 0
    start_time = time.time()
    total_steps = 0
    if use_racer:
        for outputs in model.racer_generate(
                input_ids,
                temperature=temperature,
                top_p=top_p,
                max_steps=args.max_steps,
                ac=ac
            ):
            totaltime += (time.time() - start_time)
            total_steps += 1
            output_ids = outputs["output_ids"]
            decode_ids = output_ids[0, input_len:].tolist()
            decode_ids = truncate_list(decode_ids, model.tokenizer.eos_token_id)
            text = model.tokenizer.decode(
                decode_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True
            )
            naive_text.append(
                model.tokenizer.decode(
                    output_ids[0, cur_len],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True
                )
            )
            cur_len = output_ids.shape[1]
            colored_text = highlight_text(text, naive_text, "green")
            if highlight_racer:
                history[-1][1] = colored_text
            else:
                history[-1][1] = "<span>" + html.escape(text) + "</span>"
            pure_history[-1][1] = text
            session_state["pure_history"] = pure_history
            new_tokens = cur_len - input_len
            yield history, f"{new_tokens/totaltime:.2f} tokens/s", f"{new_tokens/total_steps:.2f}", session_state
            start_time = time.time()
    else:
        for outputs in model.baseline_generate(
                input_ids,
                temperature=temperature,
                top_p=top_p,
                max_steps=args.max_steps
            ):
            totaltime += (time.time() - start_time)
            total_steps += 1
            output_ids = outputs["output_ids"]
            decode_ids = output_ids[0, input_len:].tolist()
            decode_ids = truncate_list(decode_ids, model.tokenizer.eos_token_id)
            text = model.tokenizer.decode(
                decode_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True
            )
            naive_text.append(
                model.tokenizer.decode(
                    output_ids[0, cur_len],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True
                )
            )
            cur_len = output_ids.shape[1]
            history[-1][1] = "<span>" + html.escape(text) + "</span>"
            pure_history[-1][1] = text
            session_state["pure_history"] = pure_history
            new_tokens = cur_len - input_len
            yield history, f"{new_tokens/totaltime:.2f} tokens/s", f"{new_tokens/total_steps:.2f}", session_state
            start_time = time.time()


def user(user_message, history, session_state):
    if history == None:
        history=[]
    pure_history = session_state.get("pure_history", [])
    pure_history += [[user_message, None]]
    session_state["pure_history"] = pure_history
    return "", history + [[user_message, None]], session_state


def regenerate(history, session_state):
    if not history:
        return history, None,"0.00 tokens/s","0.00",session_state
    pure_history = session_state.get("pure_history", [])
    pure_history[-1][-1] = None
    session_state["pure_history"]=pure_history
    if len(history) > 1:  # Check if there's more than one entry in history (i.e., at least one bot response)
        new_history = history[:-1]  # Remove the last bot response
        last_user_message = history[-1][0]  # Get the last user message
        return new_history + [[last_user_message, None]], None,"0.00 tokens/s","0.00", session_state
    history[-1][1] = None
    return history, None, "0.00 tokens/s","0.00", session_state


def clear(history,session_state):
    pure_history = session_state.get("pure_history", [])
    pure_history = []
    session_state["pure_history"] = pure_history
    return [], "0.00 tokens/s", "0.00", session_state


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-path",
    type=str,
    default="qwen/qwen3-1.7b",
    help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
)
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
parser.add_argument(
    "--max-steps",
    type=int,
    default=512,
    help="The maximum decoding steps.",
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

model = RacerModel.from_pretrained(
    args.model_path,
    torch_dtype=str_to_torch_dtype(args.dtype),
    low_cpu_mem_usage=True,
    device_map="auto",
    load_in_8bit=args.load_in_8bit,
    load_in_4bit=args.load_in_4bit
)

tokenizer = model.get_tokenizer()

ac = Automaton(max_nodes=args.max_nodes)
if args.max_breadth > 0:
    ac.init_logits(tokenizer.vocab_size, args.max_breadth)

warmup(model, tokenizer, ac)

custom_css = """
#speed textarea {
    color: red;   
    font-size: 30px; 
}"""

with gr.Blocks(css=custom_css) as demo:
    gs = gr.State({"pure_history": []})
    gr.Markdown('''## RACER Chatbot''')
    with gr.Row():
        speed_box = gr.Textbox(label="Speed", elem_id="speed", interactive=False, value="0.00 tokens/s")
        compression_box = gr.Textbox(label="Compression Ratio", elem_id="speed", interactive=False, value="0.00")
    with gr.Row():
        with gr.Column():
            use_racer = gr.Checkbox(label="Use RACER", value=True)
            highlight_racer = gr.Checkbox(label="Highlight the tokens generated by RACER", value=True)
        temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="temperature", value=0.7)
        top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="top_p", value=0.8)
        if temperature == 0:
            top_p = 0
    note = gr.Markdown(
        show_label=False,
        interactive=False,
        value='''The Compression Ratio is defined as the number of generated tokens divided by the number of forward passes in the original LLM. If "Highlight the tokens generated by RACER" is checked, the tokens correctly guessed by RACER 
    will be displayed in orange. Note: Checking this option may cause special formatting rendering issues in a few cases, especially when generating code'''
    )

    chatbot = gr.Chatbot(height=600, show_label=False)

    msg = gr.Textbox(label="Your input")
    with gr.Row():
        send_button = gr.Button("Send")
        stop_button = gr.Button("Stop")
        regenerate_button = gr.Button("Regenerate")
        clear_button = gr.Button("Clear")
    enter_event = msg.submit(user, [msg, chatbot,gs], [msg, chatbot,gs], queue=True).then(
        bot, [chatbot, temperature, top_p, use_racer, highlight_racer,gs], [chatbot,speed_box,compression_box,gs]
    )
    clear_button.click(clear, [chatbot,gs], [chatbot,speed_box, compression_box, gs], queue=True)

    send_event = send_button.click(user, [msg, chatbot,gs], [msg, chatbot,gs], queue=True).then(
        bot, [chatbot, temperature, top_p, use_racer, highlight_racer, gs], [chatbot, speed_box, compression_box, gs]
    )
    regenerate_event=regenerate_button.click(regenerate, [chatbot,gs], [chatbot, msg,speed_box, compression_box, gs], queue=True).then(
        bot, [chatbot, temperature, top_p, use_racer, highlight_racer,gs], [chatbot, speed_box, compression_box, gs]
    )
    stop_button.click(fn=None, inputs=None, outputs=None, cancels=[send_event, regenerate_event, enter_event])
demo.queue()
demo.launch(share=True)
