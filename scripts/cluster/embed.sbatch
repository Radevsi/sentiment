#!/usr/bin/env bash
# Supply site-specific GPU/partition/account/time/memory options at submission.
#SBATCH --job-name=finance-embed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -Eeuo pipefail
started=$SECONDS
trap 'printf "FAILED embedding line %s after %ss\n" "$LINENO" "$((SECONDS-started))" >&2' ERR
: "${FINANCE_REPO:?Set FINANCE_REPO}"
: "${FINANCE_RUN:?Set FINANCE_RUN}"
: "${FINANCE_VENV:?Set FINANCE_VENV}"
: "${HF_HOME:?Set HF_HOME to the prepared model cache on shared storage}"
cd "$FINANCE_REPO"
gpu_count=${FINANCE_GPUS:-1}
if [[ ! "$gpu_count" =~ ^[1-9][0-9]*$ ]]; then
  printf 'FAILED: FINANCE_GPUS must be a positive integer\n' >&2
  exit 2
fi
threads_per_gpu=$(( ${SLURM_CPUS_PER_TASK:-4} / gpu_count ))
if (( threads_per_gpu < 1 )); then threads_per_gpu=1; fi
export OMP_NUM_THREADS=$threads_per_gpu
devices=()
for ((gpu_index=0; gpu_index<gpu_count; gpu_index++)); do
  devices+=("cuda:$gpu_index")
done
printf '[%s] 1/1 Embed on %s GPU(s); resume saved batches\n' "$(date -Is)" "$gpu_count"
"$FINANCE_VENV/bin/python" -I -m finance_sentiment.pipeline embed --run-dir "$FINANCE_RUN" --devices "${devices[@]}" --offline --batch-size "${FINANCE_BATCH_SIZE:-64}"
printf '[%s] Embedding complete; step/overall elapsed %ss\n' "$(date -Is)" "$((SECONDS-started))"
