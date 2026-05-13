#!/usr/bin/env python3
"""
Score VQA-RAD test predictions against HF `flaviagiammarino/vqa-rad` ground truth.

Closed / Open split (standard heuristic): ground-truth answer is exactly `yes` or `no`
-> closed; otherwise open.

Metrics (defaults):
  - Closed: exact match after light normalization (VQA-RAD yes/no accuracy).
  - Open: token-level recall = |GT_tokens ∩ Pred_tokens| / |GT_tokens|  (same spirit as many med-VQA papers).
  - Overall: mean per-sample score (closed: 0/1; open: recall in [0,1]).

Some papers report *open accuracy* (exact / relaxed) instead of recall; use --open-metric exact for strict exact on open subset.

Closed-ended (GT yes/no): chat-style models rarely output only the word ``yes``/``no``.
Use ``--closed-metric yesno`` (default): extract the first standalone ``yes``/``no`` token
from the prediction (word-boundary match) and compare to GT — aligns with common VQA-RAD
practice and paper-style closed accuracy. Use ``--closed-metric exact`` for full-string match.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set

from datasets import load_dataset


def norm_text(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).lower()).replace(" ?", "?").strip()
    s = re.sub(r"[^\w\s]", "", s)  # drop punctuation for token overlap
    return s


def tokens(s: str) -> Set[str]:
    return {t for t in norm_text(s).split() if t}


def is_closed_answer(gt: str) -> bool:
    a = norm_text(gt)
    a = re.sub(r"[^\w\s]", "", a).strip()
    return a in ("yes", "no")


def closed_match_exact(gt: str, pred: str) -> float:
    g = re.sub(r"[^\w\s]", "", norm_text(gt)).strip()
    p = re.sub(r"[^\w\s]", "", norm_text(pred)).strip()
    if not g:
        return 1.0 if not p else 0.0
    return 1.0 if g == p else 0.0


def extract_yes_no(pred: str) -> Optional[str]:
    """First standalone yes/no in prediction (by start offset). None if neither."""
    t = norm_text(pred)
    yes_m = [m.start() for m in re.finditer(r"\byes\b", t)]
    no_m = [m.start() for m in re.finditer(r"\bno\b", t)]
    if yes_m and not no_m:
        return "yes"
    if no_m and not yes_m:
        return "no"
    if yes_m and no_m:
        return "yes" if min(yes_m) < min(no_m) else "no"
    return None


def closed_match_yesno(gt: str, pred: str) -> float:
    """GT must be yes/no; pred may be a full sentence."""
    g = re.sub(r"[^\w\s]", "", norm_text(gt)).strip()
    if g not in ("yes", "no"):
        return 0.0
    if closed_match_exact(gt, pred) >= 1.0:
        return 1.0
    got = extract_yes_no(pred)
    if got is None:
        return 0.0
    return 1.0 if got == g else 0.0


def open_recall(gt: str, pred: str) -> float:
    g, p = tokens(gt), tokens(pred)
    if not g:
        return 1.0 if not p else 0.0
    return len(g & p) / len(g)


def open_exact(gt: str, pred: str) -> float:
    return 1.0 if norm_text(gt) == norm_text(pred) else 0.0


def load_preds(path: str) -> Dict[str, str]:
    """Map question_id -> model output text (last line json per model_vqa.py uses key 'text')."""
    by_id: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            qid = str(o.get("question_id", ""))
            txt = o.get("text", o.get("answer", "")).strip()
            by_id[qid] = txt
    return by_id


def load_gt_from_hf(dataset_path: str, split: str = "test") -> List[dict]:
    try:
        ds = load_dataset(dataset_path, split=split)
    except Exception:
        ds = load_dataset("flaviagiammarino/vqa-rad", split=split)
    rows = []
    for i, row in enumerate(ds):
        qid = f"vqa_rad_{split}_{i}"
        rows.append(
            {
                "question_id": qid,
                "answer": row["answer"],
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", default="/media/wsqlab/data2/lqj/data/vqa-rad")
    ap.add_argument("--preds", required=True, help="answer jsonl from model_vqa.py")
    ap.add_argument("--open-metric", choices=("recall", "exact"), default="recall")
    ap.add_argument(
        "--closed-metric",
        choices=("yesno", "exact"),
        default="yesno",
        help="yesno: extract first standalone yes/no from model text (default, paper-style); "
        "exact: whole normalized string must equal yes/no",
    )
    ap.add_argument(
        "--metrics-json-out",
        default=None,
        help="If set, write machine-readable metrics (same keys as printed JSON) to this path.",
    )
    args = ap.parse_args()

    preds = load_preds(args.preds)
    gts = load_gt_from_hf(args.dataset_path, "test")

    open_fn = open_recall if args.open_metric == "recall" else open_exact
    closed_fn = closed_match_yesno if args.closed_metric == "yesno" else closed_match_exact

    n_c = n_o = 0
    sum_c = sum_o = 0.0
    sum_all = 0.0

    missing = 0
    for row in gts:
        qid = row["question_id"]
        gt = row["answer"]
        if qid not in preds:
            missing += 1
            continue
        pr = preds[qid]
        if is_closed_answer(gt):
            n_c += 1
            s = closed_fn(gt, pr)
            sum_c += s
        else:
            n_o += 1
            s = open_fn(gt, pr)
            sum_o += s
        sum_all += s

    n = n_c + n_o
    closed_acc = sum_c / n_c if n_c else float("nan")
    open_score = sum_o / n_o if n_o else float("nan")
    overall = sum_all / n if n else float("nan")

    def _pct(x: float) -> float:
        return round(x * 100, 2) if x == x else float("nan")  # x==x false for nan

    open_pct = _pct(open_score)
    closed_pct = _pct(closed_acc)
    overall_pct = _pct(overall)

    def _fmt(v: float) -> str:
        return f"{v:>6.2f}" if v == v else "   nan"

    print(
        "================================================================================\n"
        "VQA-RAD summary (paper-style columns; values are %)\n"
        f"  Open:    {_fmt(open_pct)}   (open_metric={args.open_metric})\n"
        f"  Closed:  {_fmt(closed_pct)}   (closed_metric={args.closed_metric})\n"
        f"  Overall: {_fmt(overall_pct)}\n"
        "  — Column names align with arXiv:2603.20314 Table 1 (Open / Closed / Overall).\n"
        "  — Match to paper Greedy row (e.g. 34.45 / 68.92 / 53.64) depends on same checkpoint, decoding, and metric code.\n"
        "================================================================================"
    )

    def _json_num(v: float):
        return round(v, 2) if not (isinstance(v, float) and math.isnan(v)) else None

    metrics_obj = {
                "n_test_used": n,
                "n_closed": n_c,
                "n_open": n_o,
                "Open": _json_num(open_pct),
                "Closed": _json_num(closed_pct),
                "Overall": _json_num(overall_pct),
                "closed_accuracy": _json_num(closed_pct),
                "closed_metric": args.closed_metric,
                "open_metric": args.open_metric,
                "open_score_percent": _json_num(open_pct),
                "overall_percent": _json_num(overall_pct),
                "missing_preds": missing,
            }

    print(
        json.dumps(
            metrics_obj,
            indent=2,
        )
    )

    if args.metrics_json_out:
        outp = args.metrics_json_out
        parent = os.path.dirname(os.path.abspath(outp))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as jf:
            json.dump(metrics_obj, jf, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
