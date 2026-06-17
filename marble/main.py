"""
Main entry point for running the Marble simulation engine.
"""

import argparse
import logging
import os
import sys

from marble.configs.config import Config
from marble.engine.engine import Engine


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run the Marble simulation engine.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to run the simulation with the specified config file.
    """
    args = parse_args()

    # Check if the config file exists
    if not os.path.isfile(args.config_path):
        logging.error(f"Configuration file not found: {args.config_path}")
        sys.exit(1)

    # Load configuration
    try:
        config = Config.load(args.config_path)
    except Exception as e:
        logging.error(f"Error loading configuration from {args.config_path}: {e}")
        sys.exit(1)

    # Hard override: route every agent + judge LLM call to the local Qwen3-8B
    # vLLM server, regardless of what the config files specify. The "openai/"
    # prefix is what model_prompting routes to http://localhost:9999/v1.
    # (Latent agents ignore this and use config.latent.model_name instead.)
    LOCAL_MODEL = "openai/Qwen3-8B"
    config.llm = LOCAL_MODEL
    for agent_cfg in config.agents:
        if isinstance(agent_cfg, dict):
            agent_cfg["llm"] = LOCAL_MODEL
    if isinstance(config.metrics, dict):
        config.metrics["evaluate_llm"] = {"model": LOCAL_MODEL}

    # Optional: run ANY config in graph-latent mode without editing it.
    # GRAPH_LATENT=1 flips coordinate_mode and injects a latent block, so the
    # existing graph configs (e.g. coding_configs/config_N.yaml) can be run latent
    # for an apples-to-apples comparison. Tunable via env: LATENT_MODEL /
    # LATENT_STEPS / LATENT_MAX_NEW_TOKENS / LATENT_DEVICE / LATENT_OUTPUT.
    if os.environ.get("GRAPH_LATENT") == "1":
        config.coordination_mode = "graph_latent"
        latent = config.latent if isinstance(config.latent, dict) else {}
        latent.setdefault(
            "model_name", os.environ.get("LATENT_MODEL", "/data0/xyao/models/Qwen3-8B")
        )
        latent.setdefault("latent_steps", int(os.environ.get("LATENT_STEPS", "10")))
        latent.setdefault("latent_space_realign", True)
        latent.setdefault(
            "max_new_tokens", int(os.environ.get("LATENT_MAX_NEW_TOKENS", "2048"))
        )
        latent.setdefault("device", os.environ.get("LATENT_DEVICE", "cuda"))
        # Two independent switches (default: comm on, memory compression off).
        latent.setdefault("latent_comm", os.environ.get("LATENT_COMM", "1") == "1")
        latent.setdefault(
            "latent_memory",
            os.environ.get("LATENT_MEMORY", "0") == "1",
        )
        latent.setdefault(
            "latent_memory_plan",
            os.environ.get("LATENT_MEMORY_PLAN", "0") == "1",
        )
        latent.setdefault(
            "latent_memory_max_tokens",
            int(os.environ.get("LATENT_MEMORY_MAX_TOKENS", "512")),
        )
        config.latent = latent
        if isinstance(config.output, dict):
            config.output["file_path"] = os.environ.get(
                "LATENT_OUTPUT", "result/coding_graph_latent_output.jsonl"
            )

    # Initialize and start the engine
    try:
        logging.info(f"Starting engine with configuration: {args.config_path}")
        engine = Engine(config)
        engine.start()
    except Exception:
        logging.exception(
            f"An error occurred while running the engine with configuration: {args.config_path}"
        )
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
