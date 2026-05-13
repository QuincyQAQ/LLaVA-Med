#!/usr/bin/env bash
# 一键：导出 VQA-RAD test → 跑 model_vqa → 打 Open/Closed/Overall
# 用法（在项目根或任意目录）：
#   bash scripts/vqa_rad_oneclick.sh
#   MODEL_PATH=/path/to/llava-med-v1.5-mistral-7b bash scripts/vqa_rad_oneclick.sh
# 已有预测、只想重新打分：
#   bash scripts/vqa_rad_oneclick.sh --skip-infer
# 只导出题目+图、暂不推理（无 GPU 时）：
#   bash scripts/vqa_rad_oneclick.sh --export-only
# 多卡并行（每卡一份完整模型 + 题目分片，最后合并 preds）：
#   bash scripts/vqa_rad_oneclick.sh --num-gpus 4
#   bash scripts/vqa_rad_oneclick.sh --gpus 0,1,2,7
# VGS 解码（arXiv:2603.20314，约 2× 单步推理耗时）：
#   bash scripts/vqa_rad_oneclick.sh --decoding-strategy vgs
# 实验记录（每次运行建 experiment/{策略}_{时间戳}/，并追加 experiment/result.csv）：
#   默认开启；关闭：bash scripts/vqa_rad_oneclick.sh --no-experiment  或  NO_EXPERIMENT=1

set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy 2>/dev/null || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

PYTHON="${PYTHON:-python}"

# 可按环境改
HF_VQA_RAD="${HF_VQA_RAD:-/media/wsqlab/data2/lqj/data/vqa-rad}"
WORKDIR="${WORKDIR:-/media/wsqlab/data2/lqj/data/vqa_rad_run}"
MODEL_PATH="${MODEL_PATH:-/media/wsqlab/data2/lqj/data/llava-med-v1.5-mistral-7b}"

SKIP_INFER=0
EXPORT_ONLY=0
OPEN_METRIC="recall"
CLOSED_METRIC="yesno"
NUM_GPUS=1
GPU_LIST_STR=""
DECODING_STRATEGY="${DECODING_STRATEGY:-greedy}"
VGS_SIGMA="${VGS_SIGMA:-0.07}"
VGS_POISSON_LAMBDA="${VGS_POISSON_LAMBDA:-70}"
VGS_ALPHA="${VGS_ALPHA:-1.0}"
VGS_DELTA="${VGS_DELTA:-0.01}"
NO_EXPERIMENT="${NO_EXPERIMENT:-0}"
CAPTURED_ONECLICK_CMD="$0"
for __arg in "$@"; do CAPTURED_ONECLICK_CMD+=" ${__arg}"; done
while [[ $# -gt 0 ]]; do
	case "$1" in
	--skip-infer) SKIP_INFER=1 ;;
	--export-only) EXPORT_ONLY=1 ;;
	--open-metric) OPEN_METRIC="$2"; shift ;;
	--closed-metric) CLOSED_METRIC="$2"; shift ;;
	--hf-path) HF_VQA_RAD="$2"; shift ;;
	--workdir) WORKDIR="$2"; shift ;;
	--model) MODEL_PATH="$2"; shift ;;
	--python) PYTHON="$2"; shift ;;
	--num-gpus) NUM_GPUS="$2"; shift ;;
	--gpus) GPU_LIST_STR="$2"; shift ;;
	--decoding-strategy) DECODING_STRATEGY="$2"; shift ;;
	--vgs-sigma) VGS_SIGMA="$2"; shift ;;
	--vgs-poisson-lambda) VGS_POISSON_LAMBDA="$2"; shift ;;
	--vgs-alpha) VGS_ALPHA="$2"; shift ;;
	--vgs-delta) VGS_DELTA="$2"; shift ;;
	--no-experiment) NO_EXPERIMENT=1 ;;
	-h | --help)
		echo "Usage: bash scripts/vqa_rad_oneclick.sh [options]"
		echo "  --export-only     只导出 jsonl + 图片"
		echo "  --skip-infer      跳过推理（需已有 \$WORKDIR/vqa_rad_test_preds.jsonl）"
		echo "  --open-metric recall|exact   开放题打分方式（默认 recall）"
		echo "  --closed-metric yesno|exact  封闭题：yesno=从回答里抽首个 yes/no（默认）；exact=整句须等于 yes/no"
		echo "  --hf-path PATH    HF 数据集本地目录（默认 \$HF_VQA_RAD）"
		echo "  --workdir PATH    工作目录（默认 \$WORKDIR）"
		echo "  --model PATH      模型（默认 \$MODEL_PATH）"
		echo "  --python PATH     python 解释器"
		echo "  --num-gpus N      用前 N 张物理 GPU（0..N-1）并行推理（默认 1）"
		echo "  --gpus IDS       逗号分隔物理 GPU 号，如 0,1,2,7；与 --num-gpus 二选一优先用本项"
		echo "  --decoding-strategy greedy|vgs   greedy=HF generate；vgs=VGS-Decoding arXiv:2603.20314（更慢）"
		echo "  --vgs-sigma / --vgs-poisson-lambda / --vgs-alpha / --vgs-delta   仅 vgs 时有效（默认同论文常用值）"
		echo "  --no-experiment  不写 experiment/ 快照与 result.csv（仅快速试跑）"
		echo "环境变量: MODEL_PATH, WORKDIR, HF_VQA_RAD, PYTHON, HF_HOME, DECODING_STRATEGY, VGS_*, NO_EXPERIMENT（=1 等同 --no-experiment）"
		echo "  VQA_RAD_NO_PROMPT_SUFFIX=1  导出题目不加短答后缀（与改前一致）"
		echo "  MODEL_VQA_NO_VISION_SYSTEM=1  不使用 LLaVA 默认 vision system（与改前 mistral 空 system 一致）"
		exit 0
		;;
	*) echo "未知参数: $1 （用 --help）" >&2; exit 1 ;;
	esac
	shift
done

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -lt 1 ]]; then
	echo "错误: --num-gpus 须为正整数，当前: $NUM_GPUS" >&2
	exit 1
fi
if [[ "$DECODING_STRATEGY" != "greedy" && "$DECODING_STRATEGY" != "vgs" ]]; then
	echo "错误: --decoding-strategy 须为 greedy 或 vgs，当前: $DECODING_STRATEGY" >&2
	exit 1
fi

MVQA_EXTRA=(--decoding-strategy "$DECODING_STRATEGY")
if [[ "$DECODING_STRATEGY" == "vgs" ]]; then
	MVQA_EXTRA+=(--vgs-sigma "$VGS_SIGMA" --vgs-poisson-lambda "$VGS_POISSON_LAMBDA" --vgs-alpha "$VGS_ALPHA" --vgs-delta "$VGS_DELTA")
fi

declare -a PHYS_GPUS=()
if [[ -n "$GPU_LIST_STR" ]]; then
	IFS=',' read -ra _gpu_tmp <<< "${GPU_LIST_STR// /}"
	for x in "${_gpu_tmp[@]}"; do
		[[ -n "$x" ]] || continue
		PHYS_GPUS+=("$x")
	done
	if [[ ${#PHYS_GPUS[@]} -eq 0 ]]; then
		echo "错误: --gpus 未解析出任何 GPU 号" >&2
		exit 1
	fi
elif [[ "$NUM_GPUS" -gt 1 ]]; then
	for ((i = 0; i < NUM_GPUS; i++)); do
		PHYS_GPUS+=("$i")
	done
fi
# ${#PHYS_GPUS[@]}==0 -> 单进程，不设置 CUDA_VISIBLE_DEVICES（沿用当前 shell）
# ==1 且来自 --gpus -> 绑一张卡；==1 且 NUM_GPUS 默认 -> 仍为空数组，单进程默认 cuda
# >1 -> 多进程并行

if ! "$PYTHON" -c "import datasets" 2>/dev/null; then
	echo "正在安装 datasets / tqdm …"
	"$PYTHON" -m pip install -q datasets tqdm
fi

# HF 默认缓存 ~/.cache/huggingface（常在已满的系统盘）；统一到 data2
export HF_HOME="${HF_HOME:-/media/wsqlab/data2/lqj/miniconda/hf-cache}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" 2>/dev/null || true

JSONL="$WORKDIR/vqa_rad_test.jsonl"
IMGS="$WORKDIR/images"
PREDS="$WORKDIR/vqa_rad_test_preds.jsonl"

declare -a HF_EXPORT_EXTRA=()
[[ "${VQA_RAD_NO_PROMPT_SUFFIX:-0}" == "1" ]] && HF_EXPORT_EXTRA+=(--no-prompt-suffix)

declare -a MVQA_SYSTEM_EXTRA=()
[[ "${MODEL_VQA_NO_VISION_SYSTEM:-0}" == "1" ]] && MVQA_SYSTEM_EXTRA+=(--system "")

echo "== [1/3] 导出 VQA-RAD test → jsonl + 图片"
mkdir -p "$WORKDIR"
"$PYTHON" scripts/vqa_rad_hf_export.py \
	--dataset-path "$HF_VQA_RAD" \
	--split test \
	--out-jsonl "$JSONL" \
	--out-image-dir "$IMGS" \
	"${HF_EXPORT_EXTRA[@]}"

# 实验目录：导出成功后创建（便于推理失败时仍保留目录名与时间）
EXP_RUN_DIR=""
RUN_TS=""
RUN_TIME_DISP=""
if [[ "$EXPORT_ONLY" != 1 && "$NO_EXPERIMENT" != 1 ]]; then
	EXP_ROOT="$ROOT/experiment"
	mkdir -p "$EXP_ROOT"
	RUN_TS="$(date +%Y%m%d_%H%M%S)"
	RUN_TIME_DISP="$(date '+%Y-%m-%d %H:%M:%S')"
	EXP_RUN_NAME="${DECODING_STRATEGY}_${RUN_TS}"
	EXP_RUN_DIR="$EXP_ROOT/$EXP_RUN_NAME"
	mkdir -p "$EXP_RUN_DIR"
	printf '%s\n' "$CAPTURED_ONECLICK_CMD" >"$EXP_RUN_DIR/oneclick_command.txt"
	echo "experiment: 本次运行目录 -> $EXP_RUN_DIR"
fi

if [[ "$EXPORT_ONLY" == 1 ]]; then
	echo "已按 --export-only 结束。推理示例（含 HF 缓存到 data2，避免 ~/.cache 占满系统盘）："
	echo "  export HF_HOME=/media/wsqlab/data2/lqj/miniconda/hf-cache"
	echo "  export HF_HUB_CACHE=\$HF_HOME/hub TRANSFORMERS_CACHE=\$HF_HOME/transformers"
	echo "  mkdir -p \"\$HF_HUB_CACHE\" \"\$TRANSFORMERS_CACHE\""
	echo "  cd $ROOT && PYTHONPATH=. $PYTHON llava/eval/model_vqa.py \\"
	echo "    --conv-mode mistral_instruct --model-path \"$MODEL_PATH\" \\"
	echo "    --question-file \"$JSONL\" --image-folder \"$IMGS\" \\"
	echo "    --answers-file \"$PREDS\" --temperature 0 \\"
	echo "    --decoding-strategy vgs --vgs-sigma 0.07 --vgs-poisson-lambda 70 --vgs-alpha 1.0 --vgs-delta 0.01"
	exit 0
fi

if [[ "$SKIP_INFER" != 1 ]]; then
	NW="${#PHYS_GPUS[@]}"
	if [[ "$NW" -gt 1 ]]; then
		if [[ "$DECODING_STRATEGY" == "vgs" ]]; then
			echo "== [2/3] 推理（VGS-Decoding, ${NW} 路并行；GPU: ${PHYS_GPUS[*]}）— 每步约 2× 前向，总耗时显著高于 greedy"
		else
			echo "== [2/3] 推理（Greedy, ${NW} 路并行；物理 GPU: ${PHYS_GPUS[*]}）— 每卡各加载一份模型"
		fi
	else
		if [[ "$DECODING_STRATEGY" == "vgs" ]]; then
			echo "== [2/3] 推理（VGS-Decoding arXiv:2603.20314, temperature=0）— 每步双前向，约为 greedy 数倍耗时"
		else
			echo "== [2/3] 推理（Greedy: temperature=0）— 可能较久，请保持终端别断"
		fi
	fi
	# bitsandbytes 0.41 与新版 CUDA/PyTorch 常不兼容，会在 import transformers 时崩溃；FP16 推理不需要它
	"$PYTHON" -m pip uninstall -y bitsandbytes 2>/dev/null || true

	if [[ "$NW" -gt 1 ]]; then
		rm -f "$WORKDIR"/vqa_rad_test_preds.chunk*.jsonl 2>/dev/null || true
		declare -a _pids=()
		for ((k = 0; k < NW; k++)); do
			CHUNK_PREDS="$WORKDIR/vqa_rad_test_preds.chunk${k}.jsonl"
			(
				export CUDA_VISIBLE_DEVICES="${PHYS_GPUS[$k]}"
				PYTHONPATH=. "$PYTHON" llava/eval/model_vqa.py \
					--conv-mode mistral_instruct \
					--model-path "$MODEL_PATH" \
					--question-file "$JSONL" \
					--image-folder "$IMGS" \
					--answers-file "$CHUNK_PREDS" \
					--temperature 0 \
					--num-chunks "$NW" \
					--chunk-idx "$k" \
					"${MVQA_SYSTEM_EXTRA[@]}" \
					"${MVQA_EXTRA[@]}"
			) &
			_pids+=($!)
		done
		_fail=0
		for _pid in "${_pids[@]}"; do
			if ! wait "$_pid"; then
				echo "错误: 子进程 pid=$_pid 推理失败" >&2
				_fail=1
			fi
		done
		[[ "$_fail" -eq 0 ]] || exit 1
		: >"$PREDS"
		for ((k = 0; k < NW; k++)); do
			CHUNK_PREDS="$WORKDIR/vqa_rad_test_preds.chunk${k}.jsonl"
			if [[ ! -f "$CHUNK_PREDS" ]]; then
				echo "错误: 缺少分片输出 $CHUNK_PREDS" >&2
				exit 1
			fi
			cat "$CHUNK_PREDS" >>"$PREDS"
		done
		echo "已合并 $NW 个分片 -> $PREDS"
	elif [[ "$NW" -eq 1 ]]; then
		(
			export CUDA_VISIBLE_DEVICES="${PHYS_GPUS[0]}"
			PYTHONPATH=. "$PYTHON" llava/eval/model_vqa.py \
				--conv-mode mistral_instruct \
				--model-path "$MODEL_PATH" \
				--question-file "$JSONL" \
				--image-folder "$IMGS" \
				--answers-file "$PREDS" \
				--temperature 0 \
				"${MVQA_SYSTEM_EXTRA[@]}" \
				"${MVQA_EXTRA[@]}"
		)
	else
		PYTHONPATH=. "$PYTHON" llava/eval/model_vqa.py \
			--conv-mode mistral_instruct \
			--model-path "$MODEL_PATH" \
			--question-file "$JSONL" \
			--image-folder "$IMGS" \
			--answers-file "$PREDS" \
			--temperature 0 \
			"${MVQA_SYSTEM_EXTRA[@]}" \
			"${MVQA_EXTRA[@]}"
	fi
else
	if [[ ! -f "$PREDS" ]]; then
		echo "错误: --skip-infer 但未找到 $PREDS" >&2
		exit 1
	fi
	echo "== [2/3] 跳过推理，使用已有: $PREDS"
fi

echo "== [3/3] 打分 Open / Closed / Overall"
METRICS_JSON=""
if [[ -n "$EXP_RUN_DIR" ]]; then
	METRICS_JSON="$EXP_RUN_DIR/metrics.json"
	"$PYTHON" scripts/vqa_rad_score.py \
		--dataset-path "$HF_VQA_RAD" \
		--preds "$PREDS" \
		--open-metric "$OPEN_METRIC" \
		--closed-metric "$CLOSED_METRIC" \
		--metrics-json-out "$METRICS_JSON"
else
	"$PYTHON" scripts/vqa_rad_score.py \
		--dataset-path "$HF_VQA_RAD" \
		--preds "$PREDS" \
		--open-metric "$OPEN_METRIC" \
		--closed-metric "$CLOSED_METRIC"
fi

if [[ -n "$EXP_RUN_DIR" && -f "$EXP_RUN_DIR/metrics.json" ]]; then
	_skip_infer_flag="0"
	[[ "$SKIP_INFER" == 1 ]] && _skip_infer_flag="1"
	"$PYTHON" scripts/experiment_finalize.py \
		--project-root "$ROOT" \
		--run-dir "$EXP_RUN_DIR" \
		--metrics-json "$METRICS_JSON" \
		--strategy "$DECODING_STRATEGY" \
		--run-time-display "$RUN_TIME_DISP" \
		--run-ts "$RUN_TS" \
		--preds-path "$PREDS" \
		--workdir "$WORKDIR" \
		--model-path "$MODEL_PATH" \
		--hf-path "$HF_VQA_RAD" \
		--open-metric "$OPEN_METRIC" \
		--closed-metric "$CLOSED_METRIC" \
		--skip-infer "$_skip_infer_flag" \
		--num-gpus "$NUM_GPUS" \
		--gpu-list "$GPU_LIST_STR" \
		--vgs-sigma "$VGS_SIGMA" \
		--vgs-poisson-lambda "$VGS_POISSON_LAMBDA" \
		--vgs-alpha "$VGS_ALPHA" \
		--vgs-delta "$VGS_DELTA"
fi

echo "完成。中间文件目录: $WORKDIR"
