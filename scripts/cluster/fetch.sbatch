#!/usr/bin/env bash
# Supply site-specific --partition/--account/--time/--mem at submission; see docs/cluster.md.
#SBATCH --job-name=finance-fetch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
set -Eeuo pipefail
started=$SECONDS
trap 'printf "FAILED retrieval line %s after %ss\n" "$LINENO" "$((SECONDS-started))" >&2' ERR
: "${FINANCE_REPO:?Set FINANCE_REPO to the absolute repository path}"
: "${FINANCE_RUN:?Set FINANCE_RUN to a persistent run directory}"
: "${FINANCE_VENV:?Set FINANCE_VENV to the Python environment directory}"
cd "$FINANCE_REPO"
printf '[%s] 1/1 Retrieve and filter; restart the same command to resume\n' "$(date -Is)"
"$FINANCE_VENV/bin/finance-pipeline" fetch \
  --config "${FINANCE_CONFIG:-configs/french_finance_cluster.json}" \
  --manifest "${FINANCE_MANIFEST:-manifests/fre-2012-5-all.json}" \
  --run-dir "$FINANCE_RUN" \
  --workers "${FINANCE_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
printf '[%s] Retrieval complete; step/overall elapsed %ss\n' "$(date -Is)" "$((SECONDS-started))"
