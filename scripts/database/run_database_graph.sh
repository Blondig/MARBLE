#!/bin/bash
# Run the DATABASE GRAPH benchmark on the local Qwen3-8B vLLM (text MARBLE graph
# baseline), mirroring scripts/coding/run_coding_graph.sh.
#
# The 500 configs in marble/configs/test_config_database/ are 100 distinct
# scenarios x 5 model prefixes (gpt-3.5-turbo / gpt-4o-mini / llama-3.1-70b /
# llama-3.1-8b / llama-3.3-70b), identical except the `llm` field -- which
# marble/main.py hard-overrides to openai/Qwen3-8B anyway. So we run ONE prefix =
# the 100 distinct scenarios once (the prefix is just a filename label, NOT the
# model actually used).
#
# Run from the repo root:
#     bash scripts/database/run_database_graph.sh [N]
#   N = how many scenarios (default 100). Smoke-test first:  ... 2
#   PREFIX=llama-3.1-70b bash ...   to pick a different (still Qwen3-8B) prefix.

export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:9999/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Keep the baseline text-only even if the parent shell contains latent settings.
unset GRAPH_LATENT LATENT_OUTPUT

CONFIG_DIR="marble/configs/test_config_database"
PREFIX="${PREFIX:-gpt-3.5-turbo}"
N="${1:-100}"
# DB configs write a per-scenario JSON; MARBLE_OUTPUT aggregates the whole run
# into ONE jsonl (so scripts/summarize_results.py can read it).
export MARBLE_OUTPUT="result/database_output.jsonl"

mkdir -p result
: > "${MARBLE_OUTPUT}"

i=0
for cfg in "${CONFIG_DIR}/${PREFIX}_"*.yaml; do
    [ -f "${cfg}" ] || { echo "skip: no ${CONFIG_DIR}/${PREFIX}_*.yaml"; break; }
    i=$((i + 1))
    [ "${i}" -gt "${N}" ] && break
    echo "=== DB scenario [${i}] $(basename "${cfg}") ==="
    python -m marble.main --config_path "${cfg}" || echo "WARN: run failed for ${cfg}"
done

echo "Done. Ran ${i} scenario(s).  run summaries: ${MARBLE_OUTPUT}"
