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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
printf '[%s] 1/1 Embed; resume from last saved batch\n' "$(date -Is)"
"$FINANCE_VENV/bin/finance-pipeline" embed --run-dir "$FINANCE_RUN" --device cuda --offline --batch-size "${FINANCE_BATCH_SIZE:-64}"
printf '[%s] Embedding complete; step/overall elapsed %ss\n' "$(date -Is)" "$((SECONDS-started))"
