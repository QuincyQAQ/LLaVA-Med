import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

import llava.decoding  # noqa: F401 — register decoding strategies
from llava.decoding.registry import get_decoding_fn

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle, LLAVA_VISION_SYSTEM_MESSAGE
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
import math
from transformers import set_seed, logging, StoppingCriteriaList

logging.set_verbosity_error()


def _decode_assistant_only(tokenizer, input_ids: torch.Tensor, output_ids: torch.Tensor) -> str:
    """
    HF `generate` with LLaVA's `inputs_embeds` path often returns **only** newly generated token ids,
    not the full prompt+gen sequence. If we always slice `[:, prompt_len:]`, that can be empty → empty preds.

    If `output_ids` begins with the same ids as `input_ids`, treat as full sequence and strip the prompt;
    otherwise decode the whole `output_ids` (no IMAGE_TOKEN_INDEX -200 in that tensor).
    """
    prompt_len = int(input_ids.shape[1])
    out = output_ids.to(device=input_ids.device, dtype=input_ids.dtype)
    if out.shape[1] >= prompt_len and torch.equal(out[0, :prompt_len], input_ids[0]):
        gen_ids = out[:, prompt_len:]
    else:
        gen_ids = out
    if gen_ids.numel() == 0:
        return ""
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    set_seed(0)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        if args.system != "":
            conv.system = args.system
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)[0]

        if conv.sep_style in (SeparatorStyle.TWO, SeparatorStyle.LLAMA_2):
            stop_str = conv.sep2
        else:
            stop_str = conv.sep
        keywords = [stop_str] if stop_str else []
        stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(keywords, tokenizer, input_ids)]) if keywords else None

        with torch.inference_mode():
            if args.decoding_strategy == "vgs":
                decode_fn = get_decoding_fn("vgs")
                if decode_fn is None:
                    raise RuntimeError("VGS not registered; ensure `import llava.decoding` runs.")
                output_ids = decode_fn(
                    model,
                    tokenizer,
                    input_ids,
                    image_tensor.unsqueeze(0).half().cuda(),
                    sigma=args.vgs_sigma,
                    poisson_lambda=args.vgs_poisson_lambda,
                    alpha=args.vgs_alpha,
                    delta=args.vgs_delta,
                    max_new_tokens=1024,
                    stopping_criteria=stopping_criteria,
                    temperature=args.temperature if args.temperature > 0 else 0.0,
                )
            else:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p if args.top_p is not None else 1.0,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                )

        outputs = _decode_assistant_only(tokenizer, input_ids, output_ids)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument(
        "--system",
        type=str,
        default=LLAVA_VISION_SYSTEM_MESSAGE,
        help="Conversation system message. Default: LLaVA vision assistant (same as conv_llava_llama_2). "
        "Pass an empty string to keep only the conv-mode template default (e.g. mistral_instruct uses none).",
    )
    parser.add_argument(
        "--decoding-strategy",
        type=str,
        default="greedy",
        choices=["greedy", "vgs"],
        help="greedy: HF generate. vgs: VGS-Decoding (arXiv:2603.20314), ~2× inference, no extra training.",
    )
    parser.add_argument("--vgs-sigma", type=float, default=0.07, help="Gaussian σ on CLIP input tensor (paper default)")
    parser.add_argument("--vgs-poisson-lambda", type=float, default=70.0, help="Poisson strength λ (paper default); 0=off")
    parser.add_argument("--vgs-alpha", type=float, default=1.0, help="Reweighting strength α (paper default)")
    parser.add_argument("--vgs-delta", type=float, default=0.01, help="Floor δ in max(1+α·VGS, δ) (paper default)")
    args = parser.parse_args()

    eval_model(args)
