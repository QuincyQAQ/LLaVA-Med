#!/usr/bin/env python3
"""Export Hugging Face `flaviagiammarino/vqa-rad` test split to LLaVA-Med `model_vqa.py` jsonl + images on disk."""
from __future__ import annotations

import argparse
import json
import os
import re

from datasets import load_dataset
from tqdm import tqdm

# Optional instruction suffix for VQA-style eval (improves yes/no and short-answer stability).
DEFAULT_PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."


def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).lower()).replace(" ?", "?").strip()
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-path",
        default="/media/wsqlab/data2/lqj/data/vqa-rad",
        help="Local folder from `hf download ... --local-dir` or hub id `flaviagiammarino/vqa-rad`",
    )
    ap.add_argument("--split", default="test", choices=("test", "train"))
    ap.add_argument("--out-jsonl", required=True, help="e.g. data/vqa_rad/vqa_rad_test.jsonl")
    ap.add_argument("--out-image-dir", required=True, help="e.g. data/vqa_rad/images")
    ap.add_argument(
        "--prompt-suffix",
        default=DEFAULT_PROMPT_SUFFIX,
        help="Appended to each question text (default: short-answer instruction). Set to '' to disable.",
    )
    ap.add_argument(
        "--no-prompt-suffix",
        action="store_true",
        help="Export questions only (no suffix); same as legacy behavior before this flag existed.",
    )
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out_jsonl))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.out_image_dir, exist_ok=True)

    try:
        ds = load_dataset(args.dataset_path, split=args.split)
    except Exception:
        ds = load_dataset("flaviagiammarino/vqa-rad", split=args.split)

    out = open(args.out_jsonl, "w", encoding="utf-8")
    suffix = ""
    if not args.no_prompt_suffix:
        suffix = args.prompt_suffix or ""
    for i, row in enumerate(tqdm(ds, desc=args.split)):
        img = row["image"]
        q = _norm(row["question"])
        # LLaVA-Med expects a line with image path relative to --image-folder
        rel = f"{args.split}_{i:05d}.jpg"
        dst = os.path.join(args.out_image_dir, rel)
        if hasattr(img, "save"):
            img.convert("RGB").save(dst, quality=95)
        else:
            raise RuntimeError("Row has no PIL image; check dataset version.")

        text = f"{q}{suffix}"
        rec = {
            "question_id": f"vqa_rad_{args.split}_{i}",
            "image": rel,
            "text": text,
            "ground_truth_answer": _norm(row["answer"]),
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.close()
    print("Wrote", args.out_jsonl, "and images under", args.out_image_dir)


if __name__ == "__main__":
    main()
