#!/usr/bin/env bash
# Run from the repository root. Use the cluster's Python/PyTorch modules first, if provided.
set -Eeuo pipefail
started=$SECONDS
trap 'printf "[%s] FAILED setup at line %s; total elapsed %ss\n" "$(date -Is)" "$LINENO" "$((SECONDS-started))" >&2' ERR
venv_path=${1:-.venv}
step=$SECONDS
printf '[%s] 1/3 Create environment: %s\n' "$(date -Is)" "$venv_path"
python3 -m venv --system-site-packages "$venv_path"
printf 'Step elapsed: %ss\n' "$((SECONDS-step))"
step=$SECONDS
printf '[%s] 2/3 Install retrieval tools (pip progress enabled)\n' "$(date -Is)"
"$venv_path/bin/python" -m pip install --progress-bar on -e '.[dev]'
printf 'Step elapsed: %ss\n' "$((SECONDS-step))"
step=$SECONDS
printf '[%s] 3/3 Verify command entry point\n' "$(date -Is)"
"$venv_path/bin/finance-pipeline" --help
printf 'Step elapsed: %ss; overall elapsed: %ss\n' "$((SECONDS-step))" "$((SECONDS-started))"
printf 'Activate with: source "%s/bin/activate"\n' "$venv_path"
printf 'Before embedding, install the cluster-approved GPU PyTorch build, then pip install -e ".[ml]". See docs/cluster.md.\n'
