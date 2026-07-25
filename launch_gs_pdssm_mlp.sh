#!/usr/bin/env bash
# Run pdssm-d128-iso1m + mlp on group-S dataset sweeps across 3 seeds on 8 GPUs.
set -uo pipefail

cd "$(dirname "$0")"
UV="${UV:-/home/ubuntu/.local/bin/uv}"
LOG_BASE="${LOG_BASE:-logs_gs_pdssm}"
RUN_LOG_DIR="${RUN_LOG_DIR:-run_logs/gs_pdssm_mlp}"
MAX_GPUS="${MAX_GPUS:-8}"
LAYERS=(pdssm-d128-iso1m mlp)
SEEDS=(42 43 12345)

# vocab_size:num_train_examples
DATASETS=(
  "5:100000"
  "4:3000"
  "4:50000"
  "3:250"
  "3:10000"
)

mkdir -p "$RUN_LOG_DIR"

run_job() {
  local gpu="$1" vs="$2" ntr="$3" seed="$4"
  local tag="vs${vs}_ntr${ntr}_s${seed}"
  local log_file="${RUN_LOG_DIR}/${tag}_gpu${gpu}.log"

  echo "[$(date -Is)] GPU ${gpu}: starting ${tag}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${UV}" run python -m train \
    --task group-S \
    --vocab-size "${vs}" \
    --seq-len 16 \
    --num-train-examples "${ntr}" \
    --num-test-examples 1280 \
    --layers "${LAYERS[@]}" \
    --precision 32 \
    --log-base-path "${LOG_BASE}" \
    --seed "${seed}" \
    --devices "0," \
    2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  if (( status == 0 )); then
    echo "[$(date -Is)] GPU ${gpu}: finished ${tag}"
  else
    echo "[$(date -Is)] GPU ${gpu}: FAILED ${tag} (exit ${status})" >&2
    return "${status}"
  fi
}

jobs=()
for seed in "${SEEDS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    IFS=: read -r vs ntr <<< "${ds}"
    jobs+=("${vs}:${ntr}:${seed}")
  done
done

echo "Launching ${#jobs[@]} jobs on ${MAX_GPUS} GPUs (${#DATASETS[@]} datasets x ${#SEEDS[@]} seeds)"
echo "Model layers: ${LAYERS[*]}"
echo "Logs: ${LOG_BASE}/  |  run logs: ${RUN_LOG_DIR}/"

failed=0
for ((i = 0; i < ${#jobs[@]}; i += MAX_GPUS)); do
  pids=()
  batch_end=$((i + MAX_GPUS - 1))
  if (( batch_end >= ${#jobs[@]} - 1 )); then
    batch_end=$((${#jobs[@]} - 1))
  fi
  echo "[$(date -Is)] Batch $((i / MAX_GPUS + 1)): jobs $((i + 1))-$((batch_end + 1)) of ${#jobs[@]}"

  for ((j = 0; j < MAX_GPUS && i + j < ${#jobs[@]}; j++)); do
    IFS=: read -r vs ntr seed <<< "${jobs[i + j]}"
    run_job "${j}" "${vs}" "${ntr}" "${seed}" &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=$((failed + 1))
    fi
  done
done

if (( failed > 0 )); then
  echo "Done with ${failed} failed job(s)." >&2
  exit 1
fi
echo "All ${#jobs[@]} jobs completed successfully."
