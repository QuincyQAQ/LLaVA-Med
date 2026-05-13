"""
VGS-Decoding greedy loop for LLaVA-Med (Mistral): two KV caches (original vs distorted image).

Implements Algorithm 1 from arXiv:2603.20314 at a high level:
  each step: P_orig, P_dist from two forwards; VGS; multiplicative reweight; argmax.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from llava.decoding.strategies.vgs.distortion import distort_images_clip_tensor
from llava.decoding.strategies.vgs.logits import vgs_reweighted_probs


def _kv_seq_len(past) -> int:
    """Sequence length stored in KV cache (multimodal = expanded length, not raw token count)."""
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    return int(past[0][0].shape[2])


@torch.inference_mode()
def generate_llava_med_vgs(
    model,
    tokenizer,
    input_ids: torch.LongTensor,
    images: torch.Tensor,
    *,
    sigma: float = 0.07,
    poisson_lambda: float = 70.0,
    alpha: float = 1.0,
    delta: float = 0.01,
    max_new_tokens: int = 1024,
    stopping_criteria=None,
    temperature: float = 0.0,
) -> torch.LongTensor:
    """
    Returns full token ids `[1, prompt_len + new_len]` like `model.generate`.
    """
    device = input_ids.device
    images_orig = images
    images_dist = distort_images_clip_tensor(images_orig, sigma=sigma, poisson_lambda=poisson_lambda)
    if images_orig.dtype != images_dist.dtype:
        images_dist = images_dist.to(dtype=images_orig.dtype)

    past_orig = past_dist = None
    full_ids = input_ids.clone()
    # After the first multimodal forward, KV length is the *expanded* sequence length (image patches),
    # not len(full_ids). HF's prepare_inputs_for_generation relies on attention_mask.shape[1] > len(input_ids)
    # so it can slice to only the new token; using a token-length mask re-feeds the whole prompt and breaks logits.
    attention_mask = torch.ones((1, full_ids.shape[1]), device=device, dtype=torch.long)

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else None

    for _ in range(max_new_tokens):
        if past_orig is None:
            out_o = model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                images=images_orig,
                use_cache=True,
                return_dict=True,
            )
            out_d = model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                images=images_dist,
                use_cache=True,
                return_dict=True,
            )
        else:
            mi_o = model.prepare_inputs_for_generation(
                input_ids=full_ids,
                past_key_values=past_orig,
                attention_mask=attention_mask,
                use_cache=True,
                images=images_orig,
            )
            mi_d = model.prepare_inputs_for_generation(
                input_ids=full_ids,
                past_key_values=past_dist,
                attention_mask=attention_mask,
                use_cache=True,
                images=images_dist,
            )
            out_o = model(**mi_o, return_dict=True)
            out_d = model(**mi_d, return_dict=True)

        logits_o = out_o.logits[:, -1, :]
        logits_d = out_d.logits[:, -1, :]
        past_orig = out_o.past_key_values
        past_dist = out_d.past_key_values

        p_final = vgs_reweighted_probs(logits_o, logits_d, alpha=alpha, delta=delta)
        p_final = torch.nan_to_num(p_final, nan=0.0, posinf=0.0, neginf=0.0)
        if float(p_final.sum().item()) <= 0.0:
            p_final = F.softmax(logits_o.float(), dim=-1)

        if temperature and temperature > 0:
            logits_adj = torch.log(p_final + 1e-12) / float(temperature)
            probs = F.softmax(logits_adj, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = p_final.argmax(dim=-1, keepdim=True)

        vs = getattr(tokenizer, "vocab_size", None) or int(logits_o.shape[-1])
        next_token = torch.clamp(next_token.long(), 0, max(vs - 1, 0))

        tok = int(next_token.item())
        full_ids = torch.cat([full_ids, next_token], dim=1)
        # Next step: mask length == current KV length + 1 (one query position), in expanded coordinates.
        sl = _kv_seq_len(past_orig)
        attention_mask = torch.ones((1, sl + 1), device=device, dtype=attention_mask.dtype)

        if eos_id is not None and tok == eos_id:
            break
        if pad_id is not None and tok == pad_id:
            break
        if stopping_criteria is not None:
            scores = logits_o
            if stopping_criteria(full_ids, scores):
                break

    return full_ids
