# adapted from https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_model.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_llava_kv import LlavaForConditionalGeneration as KVLlavaForConditionalGeneration
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
from .modeling_qwen3_moe_kv import Qwen3MoeForCausalLM as KVQwen3MoeForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from .modeling_qwen2_5_vl_kv import Qwen2_5_VLForConditionalGeneration as KVQwen2_5_VLForConditionalGeneration
from .modeling_qwen3_vl_kv import Qwen3VLForConditionalGeneration as KVQwen3VLForConditionalGeneration
from .modeling_openpangu_dense_kv import PanguEmbeddedForCausalLM as KVPanguEmbeddedForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from .chat_template import VICUNA_CHAT_TEMPLATE
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


_KV_MODEL_REGISTRY = {
    "LlamaForCausalLM": KVLlamaForCausalLM,
    "LlavaForConditionalGeneration": KVLlavaForConditionalGeneration,
    "Qwen2ForCausalLM": KVQwen2ForCausalLM,
    "Qwen3ForCausalLM": KVQwen3ForCausalLM,
    "Qwen3MoeForCausalLM": KVQwen3MoeForCausalLM,
    "Qwen3VLForConditionalGeneration": KVQwen3VLForConditionalGeneration,
    "Qwen2_5_VLForConditionalGeneration": KVQwen2_5_VLForConditionalGeneration,
    "MixtralForCausalLM": KVMixtralForCausalLM,
    "PanguEmbeddedForCausalLM": KVPanguEmbeddedForCausalLM,
}


class RacerModel(nn.Module):

    def __init__(
        self,
        base_model,
        base_model_name_or_path
    ):
        """
        Args:
            base_model (nn.Module): The LLM to be used.
        """
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=True)
        self.device = base_model.device
        
        arch = self.config.architectures[0]
        if "CausalLM" in arch:
            self.decoder = base_model.model
        elif "ConditionalGeneration" in arch:
            self.decoder = base_model.language_model
        else:
            raise ValueError(f"Unsupported model type: {arch}")
        
        if "vicuna" in base_model_name_or_path.lower():
            self.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
            
        if self.tokenizer.chat_template is None:
            raise NotImplementedError(f"Please add chat template for {base_model_name_or_path}")

    def get_tokenizer(self):

        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer
    
    def set_tree_mask(self, tree_mask):
        
        """Set the tree attention mask for decoding.
        """
        self.decoder.tree_mask = tree_mask

    @classmethod
    def from_pretrained(
        cls,
        base_model_path="codellama/CodeLlama-7b-instruct-hf",
        **kwargs,
    ):
        """
        Args:
            base_model_path (str): Name or path of the LLM to load.

        Returns:
            RacerModel
        """
        arch = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True).architectures[0]

        model_cls = _KV_MODEL_REGISTRY.get(arch)
        if model_cls is None:
            raise ValueError(f"Unsupported model type: {arch}")

        load_kwargs = {**kwargs, "trust_remote_code": True}
        base_model = model_cls.from_pretrained(base_model_path, **load_kwargs)

        model = cls(
            base_model,
            base_model_path,
        )

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        extra_args={}
    ):
        """Forward pass of the LLM.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.
            extra_args (dict, optional): Additional arguments for the base model.

        Returns:
            torch.Tensor: A tensor containing predictions from the LM head.
        """
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **extra_args
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])

        if output_orig:
            return outputs, orig
        raise NotImplementedError

    def racer_generate(
        self,
        input_ids,
        ac,
        ngram=10,
        temperature=0.0,
        top_p=0.8,
        max_steps=512,
        max_num_draft=64,
        max_breadth=8,
        is_draft_chain=False,
        show_accepted=False,
        debug_logits_path=None,
        extra_args={}
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.

        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data_list = self.past_key_values_data_list
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data_list,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data_list = past_key_values_data_list
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        self.set_tree_mask(None)

        # Initialize tree attention mask and process prefill tokens
        logits = initialize_logits(
            input_ids, self, past_key_values, extra_args
        )

        new_token = 0
        last_round_token = 0
        tokenizer = self.get_tokenizer()

        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        
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
            
        if show_accepted:
            accepted_texts = []

        for idx in range(max_steps):
            # Automaton will receive the next token predicted by the logit in this function
            candidates, tree_candidates, tree_attn_mask, tree_position_ids, retrieve_indices = generate_draft_tree(
                logits=logits,
                ac=ac,
                pad_token_id=pad_token_id,
                top_p=top_p,
                temperature=temperature,
                max_num_draft=max_num_draft,
                device=self.base_model.device
            )
            tree_candidates = tree_candidates[None, :]
            tree_attn_mask = tree_attn_mask[None, None, :]
            tree_position_ids = tree_position_ids[None, :]
            self.set_tree_mask(tree_attn_mask)
            logits, outputs, tree_logits = tree_decoding(
                self,
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
                eos_token_id=self.tokenizer.eos_token_id,
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
                
            text = self.tokenizer.decode(
                input_ids[0, input_len:],
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            
            if show_accepted:
                accepted_text = self.tokenizer.decode(
                    input_ids[0, -accept_length:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True
                )

                index = text.rfind(accepted_text)

                if index != -1:
                    accepted_texts.append((index, accepted_text))
                
                for index, accepted_text in reversed(accepted_texts):
                    start = text[:index]
                    end = text[index + len(accepted_text):]
                    text = start + "\033[92m" + accepted_text + "\033[0m" + end
            
            if debug_logits_path is not None:
                import json
                with open(debug_logits_path, "a", encoding="utf-8") as f:
                    position_id = input_ids.size(1) - accept_length.item() - 1
                    for token_id, token_logits in zip(input_ids[0, -accept_length-1:], tree_logits[0, retrieve_indices[best_candidate][:accept_length+1]]):
                        token_str = self.tokenizer.decode(token_id.unsqueeze(0), clean_up_tokenization_spaces=False)
                        token_logits = token_logits.detach().cpu().numpy().tolist()
                        record = {
                            "token_id": token_id.item(),
                            "position_id": position_id,
                            "token_str": token_str,
                            "logits": token_logits,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        position_id += 1

            yield {
                "text": text,
                "output_ids": input_ids
            }

            if self.tokenizer.eos_token_id == input_ids[0, -1]:
                break


    def baseline_generate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.8,
        max_steps=512,
        debug_logits_path=None,
        extra_args={}
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.

        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data_list = self.past_key_values_data_list
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data_list,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data_list = past_key_values_data_list
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        self.set_tree_mask(None)
        position_ids = torch.arange(
            input_ids.shape[1], dtype=torch.long, device=input_ids.device
        )[None, :]
        outputs = self.base_model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            position_ids=position_ids,
            **extra_args
        )
        new_token = 0
        last_round_token = 0

        for idx in range(max_steps):
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
            outputs = self.base_model(input_id, use_cache=True, position_ids=position_ids, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)

            if debug_logits_path is not None:
                import json
                with open(debug_logits_path, "a", encoding="utf-8") as f:
                    token_id = input_id[0, 0]
                    token_logits = outputs.logits[0, -1, :].detach().cpu().numpy().tolist()
                    token_str = self.tokenizer.decode(token_id.unsqueeze(0), clean_up_tokenization_spaces=False)
                    record = {
                        "token_id": token_id.item(),
                        "position_id": input_ids.size(1) - 2,
                        "token_str": token_str,
                        "logits": token_logits,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                ),
                "output_ids": input_ids
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break