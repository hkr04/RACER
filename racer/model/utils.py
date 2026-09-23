# adapted from https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/utils.py

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from . import MODEL_CLASS_MAP
import importlib


def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))



def initialize_logits(input_ids, model, past_key_values, extra_args={}):
    """
    Forward pass through the model to obtain the model outputs, and logits.


    Args:
    - input_ids (torch.Tensor): The input tensor containing token ids.
    - model: The LLM for generation.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.
    - extra_args (dict): Additional arguments to be passed to the model.

    Returns:
    - logits (torch.Tensor): logits from the LLM.
    """
    position_ids = torch.arange(
        input_ids.shape[1], dtype=torch.long, device=input_ids.device
    )[None, :]
    outputs, logits = model(
        input_ids,
        past_key_values=past_key_values,
        output_orig=True,
        position_ids=position_ids,
        extra_args=extra_args
    )
    return logits


def reset_past_key_values(passed_key_values):
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_draft_tree(logits, ac, pad_token_id, top_p=0., temperature=1., max_num_draft=64, is_draft_chain=False, device="cuda", next_token=None):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - logits (torch.Tensor): Original logits.
    - ac (Automaton): Used for retrieving candidates.
    - temperature (float): Softmax temperature for probability scaling.
    - top_p (float): Nucleus sampling threshold. If 0, greedy decoding is used.
    - max_num_draft (int): Maximum number of draft candidates to generate.
    - device (str): Device to run the computation on (default is "cuda").
    
    Returns:
    - dict: Returns candidates (paths of the tree), tree_candidates (BFS sequence of candidates),
             tree_attn_mask (attention mask for the BFS sequence), tree_position_ids (positional IDs for the BFS sequence),
             retrieve_indices (indices for reordering the logits, mapping each prefix to a BFS index).
    """

    # next_token is the residual sample from a rejected draft. Drawing again
    # from the full nucleus distribution would bias that token toward the draft.
    if next_token is not None:
        next_token = next_token.reshape(-1)[:1].to(device=logits.device)
    elif top_p == 0:
        next_token = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        assert top_p < 1, "top_p should between 0.0 and 1"
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(0)
        
    buf = ac.retrieve(next_token.item(), max_num_draft, is_draft_chain)
    
    candidates = buf.candidates
    tree_candidates = buf.tree_candidates
    tree_attn_mask = buf.attn_mask
    tree_position_ids = buf.position_ids
    retrieve_indices = buf.retrieve_indices
    
    # Keep a padding slot so the tree has at least two positions.
    if len(tree_candidates) <= 1:
        candidates = [[next_token.item(), pad_token_id]]
        tree_candidates = [next_token.item(), pad_token_id]
        tree_attn_mask = [[1, 0], [1, 1]]
        tree_position_ids = [0, 1]
        retrieve_indices = [[0, 1]]
    
    # Pad the candidates to the maximum depth with 0
    max_depth = max(len(candidate) for candidate in candidates)
    candidates = [pad_path(candidate, max_depth, pad_token_id) for candidate in candidates]
    
    # Pad the retrieved candidates to the maximum depth with -1
    seq_len = max(len(indices) for indices in retrieve_indices)
    retrieve_indices = [pad_path(indices, seq_len, -1) for indices in retrieve_indices]
    
    # Convert lists to tensors
    candidates = torch.tensor(candidates, device=device)
    tree_candidates = torch.tensor(tree_candidates, device=device)
    tree_attn_mask = torch.tensor(tree_attn_mask, device=device)
    tree_position_ids = torch.tensor(tree_position_ids, device=device)
    retrieve_indices = torch.tensor(retrieve_indices, device="cpu")  # Keep retrieve_indices on CPU for indexing operations later
    
    return candidates, tree_candidates, tree_attn_mask, tree_position_ids, retrieve_indices


def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    tree_position_ids,
    input_ids,
    retrieve_indices
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.
    
    Parameters:
    - model (nn.Module): Model to be used for decoding the tree candidates.
    - tree_candidates (torch.Tensor): Input candidates based on a tree structure.
    - past_key_values (torch.Tensor): Past states, such as key and value pairs, used in attention layers.
    - tree_position_ids (torch.Tensor): Positional IDs (Layer IDs in the Trie) of each draft token.
    - input_ids (torch.Tensor): Input sequence IDs.
    - retrieve_indices (list or torch.Tensor): Indices for reordering the logits.
    
    Returns:
    - tuple: Returns logits, and other outputs from the model.
    """

    # Compute new position IDs by adding the draft position IDs to the length of the input sequence.
    position_ids = tree_position_ids + input_ids.shape[1]

    # Use the model to decode the tree candidates. 
    # The model is expected to return each draft token's logits, and possibly other outputs.
    outputs, tree_logits = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    
    # Reorder the obtained logits based on the retrieve_indices to ensure consistency with some reference ordering.
    logits = tree_logits[0, retrieve_indices]

    return logits, outputs, tree_logits

def evaluate_posterior(
    logits, candidates, temperature, top_p=0.8, pad_token_id=None
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - pad_token_id (int): Filler id past the end of a draft path. Skipped during sampling.
    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence, not counting the root token.
    - next_token (torch.Tensor or None): Sample to use as the next root after a rejection.
      None when that root should be drawn from the unmodified target distribution.
    """
    # Greedy decoding based on temperature value
    if temperature == 0:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
            candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1).to(device=candidates.device)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, None
    elif top_p > 0:
        assert top_p < 1.0, "top_p should between 0 and 1"
        # Try each distinct sibling with acceptance probability p(x), then drop
        # its mass and renormalize. Same marginal as sampling from p directly.
        # accepted_prefix_len counts the root. The returned accept length does not.
        best_candidate_index = 0
        accepted_prefix_len = 1
        rejected_draft = False
        target_probs = None
        for position in range(1, candidates.shape[1]):
            if position != accepted_prefix_len:
                break
            rejected_draft = False
            matches_prefix = (
                candidates[:, :accepted_prefix_len]
                == candidates[best_candidate_index, :accepted_prefix_len]
            ).all(dim=1)
            prefix_row = torch.nonzero(matches_prefix, as_tuple=True)[0][0]
            token_logits = (logits[prefix_row, position - 1].unsqueeze(0) / temperature).clone()
            target_probs = F.softmax(top_p_filtering(token_logits, top_p=top_p), dim=-1)[0].clone()
            seen_token_ids = set()
            for candidate_index in range(candidates.shape[0]):
                if not bool(matches_prefix[candidate_index]):
                    continue
                draft_token_id = int(candidates[candidate_index, position])
                if (
                    draft_token_id == -1
                    or draft_token_id == pad_token_id
                    or draft_token_id in seen_token_ids
                ):
                    continue
                seen_token_ids.add(draft_token_id)
                draft_prob = float(target_probs[draft_token_id])
                # Tokens removed by nucleus filtering are not part of p.
                if draft_prob <= 0:
                    continue
                if torch.rand(()).item() <= draft_prob:
                    best_candidate_index = candidate_index
                    accepted_prefix_len += 1
                    break
                target_probs[draft_token_id] = 0
                prob_sum = target_probs.sum()
                if prob_sum > 0:
                    target_probs = target_probs / prob_sum
                rejected_draft = True
        # A full accept leaves the residual stale: it belongs to the last draft
        # position, while the next token still needs the unmodified distribution.
        next_token = None
        if (
            rejected_draft
            and accepted_prefix_len != candidates.shape[1]
            and target_probs is not None
            and target_probs.sum().item() > 0
        ):
            next_token = torch.multinomial(target_probs, 1).view(-1)
        best_candidate = torch.tensor(
            best_candidate_index, dtype=torch.long, device=candidates.device
        )
        accept_length = torch.tensor(
            accepted_prefix_len - 1, dtype=torch.long, device=candidates.device
        )
        return best_candidate, accept_length, next_token
    else:
        raise NotImplementedError


def update_inference_inputs(
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
    eos_token_id,
):
    """
    Update the input sequences and relevant tensors based on the selected best candidate from the inference results.

    Args:
    - input_ids (torch.Tensor): Current input token sequences.
    - candidates (torch.Tensor): Candidate token sequences generated in the current step.
    - best_candidate (int): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - retrieve_indices (torch.Tensor): Indices to map tree to a cartesian product.
    - outputs, logits (torch.Tensor): Model's outputs from the previous inference step.
    - new_token (int): Counter for the new tokens added during inference.
    - past_key_values_data_list (list): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.
    - eos_token_id (int): End-of-sequence token ID, used to determine when generation should stop.

    Returns:
    - input_ids (torch.Tensor): Updated input token sequences.
    - logits (torch.Tensor): Updated logits.
    - new_token (int): Updated counter for the new tokens added.
    - accept_length (int): Real accept length after possible truncation by eos_token_id
    """
    # Check if eos_token_id is in the accepted candidate sequence
    # if so, truncate the sequence at the eos_token_id
    eos_index = [
        i
        for i, id in enumerate(candidates[best_candidate, : accept_length + 1])
        if id == eos_token_id
    ]

    if eos_index:
        accept_length = eos_index[0]
    
    # Calculate the starting position for new tokens based on the previous input length
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [
            input_ids,
            candidates[None, best_candidate, : accept_length + 1].to(input_ids.device),
        ],
        dim=-1,
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based
    # on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[
            ..., select_indices.to(past_key_values_data.device), :
        ]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[
            ..., prev_input_len : prev_input_len + tgt.shape[-2], :
        ]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    # Extract logits for the accepted tokens
    logits = logits[None, best_candidate, accept_length : accept_length + 1]

    # Update the new token counter
    new_token += accept_length + 1

    return input_ids, logits, new_token, accept_length


def top_p_filtering(logits, top_p=0.0, filter_value=float('-inf')):
    # from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79


    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
    return logits