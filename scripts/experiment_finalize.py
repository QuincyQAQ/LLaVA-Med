#!/usr/bin/env python3
"""
After vqa_rad_oneclick.sh: copy code/preds snapshot, write per-run CSV, append experiment/result.csv.

Recovers implementation when main repo is overwritten: see run_manifest.json + code_snapshot/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _git_head(project_root: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


def _read_metrics(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--metrics-json", required=True)
    ap.add_argument("--strategy", required=True, help="greedy | vgs")
    ap.add_argument("--run-time-display", required=True, help='e.g. "2026-05-12 14:30:22"')
    ap.add_argument("--run-ts", required=True, help="folder timestamp suffix, e.g. 20260512_143022")
    ap.add_argument("--preds-path", default="", help="copy preds jsonl into run dir")
    ap.add_argument("--workdir", default="")
    ap.add_argument("--model-path", default="")
    ap.add_argument("--hf-path", default="")
    ap.add_argument("--open-metric", default="recall")
    ap.add_argument("--closed-metric", default="yesno")
    ap.add_argument("--skip-infer", default="0")
    ap.add_argument("--num-gpus", default="1")
    ap.add_argument("--gpu-list", default="")
    ap.add_argument("--vgs-sigma", default="")
    ap.add_argument("--vgs-poisson-lambda", default="")
    ap.add_argument("--vgs-alpha", default="")
    ap.add_argument("--vgs-delta", default="")
    ap.add_argument("--oneclick-command", default="", help="full bash command line for reproducibility")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    metrics_path = Path(args.metrics_json).resolve()
    if not metrics_path.is_file():
        print(f"experiment_finalize: metrics file missing: {metrics_path}", file=sys.stderr)
        sys.exit(1)

    m = _read_metrics(metrics_path)
    open_v = m.get("Open")
    closed_v = m.get("Closed")
    overall_v = m.get("Overall")

    cmdline = (args.oneclick_command or "").strip()
    if not cmdline:
        p_cmd = run_dir / "oneclick_command.txt"
        if p_cmd.is_file():
            cmdline = p_cmd.read_text(encoding="utf-8").strip()

    code_dst = run_dir / "code_snapshot"
    code_dst.mkdir(parents=True, exist_ok=True)

    # Always snapshot evaluation + orchestration entrypoints
    for rel in (
        "llava/eval/model_vqa.py",
        "scripts/vqa_rad_score.py",
        "scripts/vqa_rad_oneclick.sh",
        "scripts/vqa_rad_hf_export.py",
        "scripts/experiment_finalize.py",
    ):
        src = root / rel
        if src.is_file():
            _copy_file(src, code_dst / Path(rel).name)

    if args.strategy == "vgs":
        vgs_src = root / "llava" / "decoding" / "strategies" / "vgs"
        if vgs_src.is_dir():
            _copy_tree(vgs_src, code_dst / "vgs")
        # small registry glue (optional but helps if vgs package imports change)
        reg = root / "llava" / "decoding" / "registry.py"
        if reg.is_file():
            _copy_file(reg, code_dst / "decoding_registry.py")
        dec_init = root / "llava" / "decoding" / "__init__.py"
        if dec_init.is_file():
            _copy_file(dec_init, code_dst / "decoding_package_init.py")
        strat_init = root / "llava" / "decoding" / "strategies" / "__init__.py"
        if strat_init.is_file():
            _copy_file(strat_init, code_dst / "strategies_package_init.py")

    preds_path = Path(args.preds_path) if args.preds_path else None
    if preds_path and preds_path.is_file():
        _copy_file(preds_path, run_dir / "preds.jsonl")

    manifest: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "strategy": args.strategy,
        "run_ts": args.run_ts,
        "run_time_display": args.run_time_display,
        "model_path": args.model_path,
        "workdir": args.workdir,
        "hf_dataset_path": args.hf_path,
        "open_metric": args.open_metric,
        "closed_metric": args.closed_metric,
        "skip_infer": args.skip_infer == "1",
        "num_gpus": args.num_gpus,
        "gpu_list": args.gpu_list,
        "metrics": {"Open": open_v, "Closed": closed_v, "Overall": overall_v, **{k: m[k] for k in ("n_test_used", "n_closed", "n_open", "missing_preds") if k in m}},
        "git_commit": _git_head(root),
        "oneclick_command": cmdline,
        "python": sys.version.split()[0],
        "finalized_at": datetime.now().isoformat(timespec="seconds"),
    }
    if args.strategy == "vgs":
        manifest["vgs"] = {
            "sigma": args.vgs_sigma,
            "poisson_lambda": args.vgs_poisson_lambda,
            "alpha": args.vgs_alpha,
            "delta": args.vgs_delta,
        }

    with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Per-run CSV: same basename as run folder
    csv_name = f"{args.strategy}_{args.run_ts}.csv"
    csv_path = run_dir / csv_name
    fieldnames = ["策略名", "Open", "Closed", "Overall", "运行时间", "运行目录", "open_metric", "closed_metric", "git_commit", "备注"]
    note = f"metrics_json={metrics_path.name}; preds_copied={'yes' if preds_path and preds_path.is_file() else 'no'}"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(
            {
                "策略名": args.strategy,
                "Open": "" if open_v is None else str(open_v),
                "Closed": "" if closed_v is None else str(closed_v),
                "Overall": "" if overall_v is None else str(overall_v),
                "运行时间": args.run_time_display,
                "运行目录": str(run_dir),
                "open_metric": args.open_metric,
                "closed_metric": args.closed_metric,
                "git_commit": manifest.get("git_commit", ""),
                "备注": note,
            }
        )

    # Aggregate result.csv under experiment/
    exp_root = root / "experiment"
    exp_root.mkdir(parents=True, exist_ok=True)
    result_csv = exp_root / "result.csv"
    agg_fields = ["策略名", "Open", "Closed", "Overall", "运行时间", "运行目录", "open_metric", "closed_metric", "git_commit", "备注"]
    row = {
        "策略名": args.strategy,
        "Open": "" if open_v is None else str(open_v),
        "Closed": "" if closed_v is None else str(closed_v),
        "Overall": "" if overall_v is None else str(overall_v),
        "运行时间": args.run_time_display,
        "运行目录": str(run_dir),
        "open_metric": args.open_metric,
        "closed_metric": args.closed_metric,
        "git_commit": manifest.get("git_commit", ""),
        "备注": note,
    }
    write_header = not result_csv.is_file()
    with open(result_csv, "a", encoding="utf-8-sig", newline="") as rf:
        w = csv.DictWriter(rf, fieldnames=agg_fields)
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"experiment: snapshot + CSV -> {run_dir}")
    print(f"experiment: appended row -> {result_csv}")


if __name__ == "__main__":
    main()
