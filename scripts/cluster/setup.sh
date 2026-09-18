#!/usr/bin/env bash
# Run from the repository root using the cluster's Python 3.10+ interpreter.
set -Eeuo pipefail
# Keep inherited user packages and Python path overrides out of setup and its subprocesses.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
started=$SECONDS
trap 'printf "[%s] FAILED setup at line %s; total elapsed %ss\n" "$(date -Is)" "$LINENO" "$((SECONDS-started))" >&2' ERR
venv_path=${1:-.venv}
step=$SECONDS
printf '[%s] 1/4 Create/update isolated environment: %s\n' "$(date -Is)" "$venv_path"
# Reusing the path also disables system-site-packages in older environments.
# No --clear: preserve packages already installed inside this environment.
python3 -I -m venv "$venv_path"
printf 'Step elapsed: %ss\n' "$((SECONDS-step))"
step=$SECONDS
printf '[%s] 2/4 Install retrieval tools (pip progress enabled)\n' "$(date -Is)"
"$venv_path/bin/python" -I -m pip install --progress-bar on -e .
printf 'Step elapsed: %ss\n' "$((SECONDS-step))"
step=$SECONDS
printf '[%s] 3/4 Check dependency consistency\n' "$(date -Is)"
"$venv_path/bin/python" -I -m pip check
printf 'Step elapsed: %ss\n' "$((SECONDS-step))"
step=$SECONDS
printf '[%s] 4/4 Verify command entry point\n' "$(date -Is)"
"$venv_path/bin/python" -I -m finance_sentiment.pipeline --help
printf 'Step elapsed: %ss; overall elapsed: %ss\n' "$((SECONDS-step))" "$((SECONDS-started))"
printf 'Activate with: source "%s/bin/activate"\n' "$venv_path"
printf 'Before embedding, install the cluster-approved GPU PyTorch build, then pip install -e ".[ml]". See docs/cluster.md.\n'
