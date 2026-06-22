#!/bin/bash
# Run the BARGAINING GRAPH benchmark on the local Qwen3-8B vLLM, mirroring the
# coding/research graph runners.
#
# Expected configs (generated from multiagentbench/bargaining/bargaining_main.jsonl):
#   multiagentbench/bargaining_yaml_files/task_1.yaml ... task_100.yaml
#
# Run from the repo root:
#   bash scripts/bargaining/run_bargaining_graph.sh [N]
# N defaults to 100. Override the config directory with BARGAINING_CONFIG_DIR.

export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:9999/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Keep the baseline text-only even if the parent shell contains latent settings.
unset GRAPH_LATENT LATENT_OUTPUT

CONFIG_DIR="${BARGAINING_CONFIG_DIR:-multiagentbench/bargaining_yaml_files}"
N="${1:-100}"
export MARBLE_OUTPUT="result/bargaining_graph_solutions.jsonl"

if [ ! -d "${CONFIG_DIR}" ]; then
    echo "ERROR: bargaining config directory not found: ${CONFIG_DIR}"
    echo "Expected generated files such as ${CONFIG_DIR}/task_1.yaml"
    exit 1
fi

mkdir -p result
: > "${MARBLE_OUTPUT}"

ran=0
for id in $(seq 1 "${N}"); do
    CONFIG_FILE="${CONFIG_DIR}/task_${id}.yaml"
    [ -f "${CONFIG_FILE}" ] || { echo "skip: no ${CONFIG_FILE}"; continue; }
    echo "=== Bargaining task ID=${id} ==="
    python -m marble.main --config_path "${CONFIG_FILE}" || echo "WARN: run failed for ID=${id}"
    ran=$((ran + 1))
done

echo "Done. Ran ${ran} bargaining task(s).  summaries: ${MARBLE_OUTPUT}"
