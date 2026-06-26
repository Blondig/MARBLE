#!/bin/bash
# Run the MINECRAFT benchmark in GRAPH-LATENT mode on the local Qwen3-8B vLLM,
# mirroring scripts/database/run_database_graph_latent.sh so it's directly
# comparable to run_minecraft_graph.sh (the text baseline).
#
# REQUIRES a running Minecraft server; HEAVY (max_iterations=20). Smoke first:
#   LATENT_MEMORY=1 LATENT_MEMORY_PLAN=1 LATENT_MEMORY_MAX_TOKENS=256 \
#     bash scripts/minecraft/run_minecraft_graph_latent.sh 10
#
# Same configs as the baseline (one prefix = 100 distinct scenarios). GRAPH_LATENT=1
# makes marble/main.py flip coordinate_mode -> graph_latent and inject the latent
# block; model is force-overridden to the local Qwen3-8B. Switches (same as coding):
#   LATENT_STEPS (10) / LATENT_COMM (on) / LATENT_MEMORY / LATENT_MEMORY_PLAN /
#   LATENT_MEMORY_MAX_TOKENS (512).

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
export MARBLE_OUTPUT="result/minecraft_graph_latent_steps${LATENT_STEPS}${_TAG}.jsonl"

CONFIG_DIR="marble/configs/test_config_minecraft"
PREFIX="${PREFIX:-gpt-35-turbo}"
N="${1:-100}"

mkdir -p result
: > "${MARBLE_OUTPUT}"

i=0
for cfg in "${CONFIG_DIR}/test_config_${PREFIX}_"*.yaml; do
    [ -f "${cfg}" ] || { echo "skip: no ${CONFIG_DIR}/test_config_${PREFIX}_*.yaml"; break; }
    i=$((i + 1))
    [ "${i}" -gt "${N}" ] && break
    echo "=== [graph-latent] Minecraft scenario [${i}] $(basename "${cfg}") (latent_steps=${LATENT_STEPS}) ==="
    python -m marble.main --config_path "${cfg}" || echo "WARN: run failed for ${cfg}"
done

echo "Done. Ran ${i} scenario(s).  summaries: ${MARBLE_OUTPUT}"
