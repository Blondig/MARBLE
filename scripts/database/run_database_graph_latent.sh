#!/bin/bash
# Run the DATABASE benchmark in GRAPH-LATENT mode on the local Qwen3-8B vLLM (the
# LatentMAS analogue), mirroring scripts/coding/run_coding_graph_latent.sh so it's
# directly comparable to run_database_graph.sh (the text baseline).
#
# Reuses the SAME configs as the baseline (one model prefix = 100 distinct
# scenarios -- see run_database_graph.sh on the 5x duplication). GRAPH_LATENT=1
# makes marble/main.py flip coordinate_mode -> graph_latent and inject the latent
# block at runtime, so NO variant config files are created. Model is hard-
# overridden to the local Qwen3-8B in main.py.
#
# Run from the repo root (transformers loads Qwen3-8B for the agents; vLLM:9999
# must be up for the planning judge / tool calls):
#     bash scripts/database/run_database_graph_latent.sh [N]
#     LATENT_MEMORY=1 LATENT_MEMORY_PLAN=1 LATENT_MEMORY_MAX_TOKENS=256 \
#       bash scripts/database/run_database_graph_latent.sh 100
#   N = how many scenarios (default 100). Switches (same as coding):
#     LATENT_STEPS (default 10), LATENT_COMM (default on), LATENT_MEMORY,
#     LATENT_MEMORY_PLAN, LATENT_MEMORY_MAX_TOKENS (default 512).

export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:9999/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export GRAPH_LATENT=1
export LATENT_STEPS="${LATENT_STEPS:-10}"

# Tag the output by which switches are on (mirrors run_coding_graph_latent.sh) so
# comm-only / memory / plan variants don't mix into one jsonl.
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
# DB configs write a per-scenario JSON; MARBLE_OUTPUT aggregates the whole run
# into ONE jsonl (overrides the GRAPH_LATENT default path too).
export MARBLE_OUTPUT="result/database_graph_latent_steps${LATENT_STEPS}${_TAG}.jsonl"

CONFIG_DIR="marble/configs/test_config_database"
PREFIX="${PREFIX:-gpt-3.5-turbo}"
N="${1:-100}"

mkdir -p result
: > "${MARBLE_OUTPUT}"

i=0
for cfg in "${CONFIG_DIR}/${PREFIX}_"*.yaml; do
    [ -f "${cfg}" ] || { echo "skip: no ${CONFIG_DIR}/${PREFIX}_*.yaml"; break; }
    i=$((i + 1))
    [ "${i}" -gt "${N}" ] && break
    echo "=== [graph-latent] DB scenario [${i}] $(basename "${cfg}") (latent_steps=${LATENT_STEPS}) ==="
    python -m marble.main --config_path "${cfg}" || echo "WARN: run failed for ${cfg}"
done

echo "Done. Ran ${i} scenario(s).  summaries: ${MARBLE_OUTPUT}"
