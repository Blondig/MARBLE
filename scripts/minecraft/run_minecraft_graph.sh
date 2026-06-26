#!/bin/bash
# Run the MINECRAFT GRAPH benchmark on the local Qwen3-8B vLLM (text MARBLE graph
# baseline), mirroring scripts/database/run_database_graph.sh.
#
# REQUIRES a running Minecraft server (the Minecraft env connects to it). This is
# HEAVY: configs use max_iterations=20 per task. Smoke a few first:
#     bash scripts/minecraft/run_minecraft_graph.sh 10
#
# The 500 configs in marble/configs/test_config_minecraft/ are 100 scenarios x 5
# model prefixes (gpt-35-turbo / gpt-4o-mini / llama-31-70b / llama-31-8b /
# llama-33-70b), identical except the `llm` field -- which main.py force-overrides
# to openai/Qwen3-8B. So we run ONE prefix = 100 distinct scenarios (prefix is just
# a filename label, NOT the model used).
#
# Run from the repo root:
#     bash scripts/minecraft/run_minecraft_graph.sh [N]       (default 100)
#   PREFIX=llama-31-70b bash ...   to pick a different (still Qwen3-8B) prefix.

export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:9999/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Keep the baseline text-only even if the parent shell contains latent settings.
unset GRAPH_LATENT LATENT_OUTPUT

CONFIG_DIR="marble/configs/test_config_minecraft"
PREFIX="${PREFIX:-gpt-35-turbo}"
N="${1:-100}"
# Aggregate the whole run into ONE jsonl (so scripts/summarize_results.py can read it).
export MARBLE_OUTPUT="result/minecraft_output.jsonl"

mkdir -p result
: > "${MARBLE_OUTPUT}"

i=0
for cfg in "${CONFIG_DIR}/test_config_${PREFIX}_"*.yaml; do
    [ -f "${cfg}" ] || { echo "skip: no ${CONFIG_DIR}/test_config_${PREFIX}_*.yaml"; break; }
    i=$((i + 1))
    [ "${i}" -gt "${N}" ] && break
    echo "=== Minecraft scenario [${i}] $(basename "${cfg}") ==="
    python -m marble.main --config_path "${cfg}" || echo "WARN: run failed for ${cfg}"
done

echo "Done. Ran ${i} scenario(s).  run summaries: ${MARBLE_OUTPUT}"
