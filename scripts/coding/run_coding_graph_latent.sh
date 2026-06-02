#!/bin/bash
# Run ALL coding configs in GRAPH-LATENT mode (the LatentMAS analogue), mirroring
# scripts/coding/run_coding_graph.sh (the text graph baseline) so the two are
# directly comparable.
#
# Reuses the existing per-task configs marble/configs/coding_configs/config_N.yaml
# (same task/agents/relationships as the text run); GRAPH_LATENT=1 makes
# marble/main.py flip coordinate_mode -> graph_latent and inject the latent block
# at runtime, so NO variant config files are created. The model is also hard-
# overridden to the local Qwen3-8B in main.py.
#
# Run from the repo root (transformers loads Qwen3-8B for agents; vLLM:9999 must
# be up for the planning judge):
#     bash scripts/coding/run_coding_graph_latent.sh [N]
#     LATENT_STEPS=40 bash scripts/coding/run_coding_graph_latent.sh 5
#   N = how many tasks (default 100). LATENT_STEPS tunes latent depth (try 10, 40).

export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:9999/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export GRAPH_LATENT=1
export LATENT_STEPS="${LATENT_STEPS:-10}"
export LATENT_OUTPUT="result/coding_graph_latent_steps${LATENT_STEPS}.jsonl"

WORKSPACE_DIR="workspace"
CONFIG_DIR="marble/configs/coding_configs"
SOLU_DIR="marble/logs/qwen3-8b/coding_latent_steps${LATENT_STEPS}"
N="${1:-100}"

mkdir -p "${WORKSPACE_DIR}" "${SOLU_DIR}" result

for id in $(seq 1 "${N}"); do
    CONFIG_FILE="${CONFIG_DIR}/config_${id}.yaml"
    [ -f "${CONFIG_FILE}" ] || { echo "skip: no ${CONFIG_FILE}"; continue; }
    echo "=== [graph-latent] coding ID=${id} (latent_steps=${LATENT_STEPS}) ==="
    rm -rf "${WORKSPACE_DIR:?}"/*
    python -m marble.main --config_path "${CONFIG_FILE}" || echo "WARN: run failed for ID=${id}"
    if [ -f "${WORKSPACE_DIR}/solution.py" ]; then
        cp "${WORKSPACE_DIR}/solution.py" "${SOLU_DIR}/solution_${id}.py"
    else
        echo "WARN: no solution.py for ID=${id}"
    fi
done

echo "Done. Solutions: ${SOLU_DIR}/  |  summaries: ${LATENT_OUTPUT}"
