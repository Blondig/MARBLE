#!/bin/bash
# Run the RESEARCH GRAPH benchmark on the local Qwen3-8B vLLM, mirroring
# scripts/coding/run_coding_graph.sh.
#
# Expected configs: the FAITHFUL anonymous configs (agent_id + bio, exactly as the
# original benchmark ran -- no real names; see base_agent.py prompt 'You are
# "agent1": "<bio>"'), generated from research_main.jsonl by jsonl2yaml:
#   multiagentbench/research_yaml_files/task_1.yaml ... task_100.yaml
#
# NOTE: the name-restored variant (research_yaml_files_named/, from
# restore_profile_names.py) is a DEVIATION from the original -- use it only as an
# optional experiment via RESEARCH_CONFIG_DIR=multiagentbench/research_yaml_files_named.
#
# Run from the repo root:
#   bash scripts/research/run_research_graph.sh [N]
# N defaults to 100. Override the config directory with RESEARCH_CONFIG_DIR.

export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:9999/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Keep this baseline text-only even if the parent shell previously exported a
# latent-run switch.
unset GRAPH_LATENT LATENT_OUTPUT

CONFIG_DIR="${RESEARCH_CONFIG_DIR:-multiagentbench/research_yaml_files}"
N="${1:-100}"
export MARBLE_OUTPUT="result/research_graph_solutions.jsonl"

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
    echo "=== Research task ID=${id} ==="
    python -m marble.main --config_path "${CONFIG_FILE}" || echo "WARN: run failed for ID=${id}"
    ran=$((ran + 1))
done

echo "Done. Ran ${ran} research task(s).  summaries: ${MARBLE_OUTPUT}"
