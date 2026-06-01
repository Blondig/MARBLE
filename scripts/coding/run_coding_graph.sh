#!/bin/bash
# Run the coding GRAPH benchmark on the local Qwen3-8B vLLM, producing one
# solution.py per task.
#
# Uses the pre-generated per-task configs in marble/configs/coding_configs/
# (config_N.yaml == benchmark.jsonl task N, verified), so it skips the fragile
# update_coding_config step entirely.
#
# IMPORTANT: graph mode does NOT score coding — confirmed in this repo AND in
# the upstream ulab-uiuc/MARBLE (only star_coordinate calls evaluate_code_quality;
# graph has no Coding branch). The paper says coding uses rule-based metrics
# (Appendix A.7), but neither the test cases nor a rule-based scorer are shipped.
# So this script ONLY produces the solution.py artifacts; scoring is a separate
# step to decide later.
#
# Run from the repo root:
#     bash scripts/coding/run_coding_graph.sh [N]
#   N = how many tasks to run (default 100). Smoke-test one first:  ... 1
#
# Model is hard-overridden to openai/Qwen3-8B inside marble/main.py, so the
# together_ai/gpt model names in the configs are ignored.

export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:9999/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

WORKSPACE_DIR="workspace"                          # matches workspace_dir in the configs
CONFIG_DIR="marble/configs/coding_configs"
SOLU_DIR="marble/logs/qwen3-8b/coding_solutions"
N="${1:-100}"

mkdir -p "${WORKSPACE_DIR}" "${SOLU_DIR}" result

for id in $(seq 1 "${N}"); do
    CONFIG_FILE="${CONFIG_DIR}/config_${id}.yaml"
    [ -f "${CONFIG_FILE}" ] || { echo "skip: no ${CONFIG_FILE}"; continue; }
    echo "=== Coding task ID=${id} ==="
    rm -rf "${WORKSPACE_DIR:?}"/*
    python -m marble.main --config_path "${CONFIG_FILE}" || echo "WARN: run failed for ID=${id}"
    if [ -f "${WORKSPACE_DIR}/solution.py" ]; then
        cp "${WORKSPACE_DIR}/solution.py" "${SOLU_DIR}/solution_${id}.py"
        echo "saved ${SOLU_DIR}/solution_${id}.py"
    else
        echo "WARN: no solution.py produced for ID=${id}"
    fi
done

echo "Done. Solutions: ${SOLU_DIR}/  |  run summaries: result/development_output.jsonl"
