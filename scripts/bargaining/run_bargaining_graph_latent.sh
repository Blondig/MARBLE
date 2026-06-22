#!/bin/bash
# Run the BARGAINING benchmark in GRAPH-LATENT mode on the same generated
# configs as run_bargaining_graph.sh. The bargaining tools and evaluator stay
# unchanged; GRAPH_LATENT only switches the gated communication/memory paths.
#
# Run from the repo root:
#   bash scripts/bargaining/run_bargaining_graph_latent.sh [N]
#   LATENT_MEMORY=1 LATENT_MEMORY_PLAN=1 LATENT_MEMORY_MAX_TOKENS=256 \
#     bash scripts/bargaining/run_bargaining_graph_latent.sh 100
#
# N defaults to 100. Supported switches match the coding runner:
# LATENT_STEPS, LATENT_COMM, LATENT_MEMORY, LATENT_MEMORY_PLAN, and
# LATENT_MEMORY_MAX_TOKENS. Override configs with BARGAINING_CONFIG_DIR.

export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:9999/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export GRAPH_LATENT=1
export LATENT_STEPS="${LATENT_STEPS:-10}"

_TAG=""
if [ "${LATENT_MEMORY:-0}" = "1" ]; then
    _TAG="${_TAG}_memory${LATENT_MEMORY_MAX_TOKENS:-512}"
fi
if [ "${LATENT_MEMORY_PLAN:-0}" = "1" ]; then
    if [ "${LATENT_MEMORY:-0}" = "1" ]; then
        _TAG="${_TAG}_plan"
    else
        _TAG="${_TAG}_plan${LATENT_MEMORY_MAX_TOKENS:-512}"
    fi
fi
[ "${LATENT_COMM:-1}" = "0" ] && _TAG="${_TAG}_nocomm"

CONFIG_DIR="${BARGAINING_CONFIG_DIR:-multiagentbench/bargaining_yaml_files}"
N="${1:-100}"
export MARBLE_OUTPUT="result/bargaining_graph_latent_steps${LATENT_STEPS}${_TAG}.jsonl"

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
    echo "=== [graph-latent] Bargaining task ID=${id} (latent_steps=${LATENT_STEPS}) ==="
    python -m marble.main --config_path "${CONFIG_FILE}" || echo "WARN: run failed for ID=${id}"
    ran=$((ran + 1))
done

echo "Done. Ran ${ran} bargaining task(s).  summaries: ${MARBLE_OUTPUT}"
