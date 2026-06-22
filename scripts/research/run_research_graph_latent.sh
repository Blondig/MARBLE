#!/bin/bash
# Run the RESEARCH benchmark in GRAPH-LATENT mode on the same generated configs
# as run_research_graph.sh, preserving the original research task/agent graph.
#
# Run from the repo root:
#   bash scripts/research/run_research_graph_latent.sh [N]
#   LATENT_MEMORY=1 LATENT_MEMORY_PLAN=1 LATENT_MEMORY_MAX_TOKENS=256 \
#     bash scripts/research/run_research_graph_latent.sh 100
#
# N defaults to 100. Supported switches match the coding runner:
# LATENT_STEPS, LATENT_COMM, LATENT_MEMORY, LATENT_MEMORY_PLAN, and
# LATENT_MEMORY_MAX_TOKENS. Override configs with RESEARCH_CONFIG_DIR.

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

CONFIG_DIR="${RESEARCH_CONFIG_DIR:-multiagentbench/research_yaml_files_named}"
N="${1:-100}"
export MARBLE_OUTPUT="result/research_graph_latent_steps${LATENT_STEPS}${_TAG}.jsonl"

if [ ! -d "${CONFIG_DIR}" ]; then
    echo "ERROR: research config directory not found: ${CONFIG_DIR}"
    echo "Expected generated files such as ${CONFIG_DIR}/task_1.yaml"
    exit 1
fi

mkdir -p result
: > "${MARBLE_OUTPUT}"

ran=0
for id in $(seq 1 "${N}"); do
    CONFIG_FILE="${CONFIG_DIR}/task_${id}.yaml"
    [ -f "${CONFIG_FILE}" ] || { echo "skip: no ${CONFIG_FILE}"; continue; }
    echo "=== [graph-latent] Research task ID=${id} (latent_steps=${LATENT_STEPS}) ==="
    python -m marble.main --config_path "${CONFIG_FILE}" || echo "WARN: run failed for ID=${id}"
    ran=$((ran + 1))
done

echo "Done. Ran ${ran} research task(s).  summaries: ${MARBLE_OUTPUT}"
