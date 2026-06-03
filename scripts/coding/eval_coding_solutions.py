#!/usr/bin/env python3
"""Offline code-quality scoring for already-generated coding solutions.

Reuses ``Evaluator.evaluate_code_quality`` (the LLM judge MARBLE ships: 4 criteria
-- instruction_following / executability / consistency / quality, each 1-5) on the
saved ``solution_<id>.py`` files, pairing each with its per-task
``config_<id>.yaml`` (its ``task.content``). This scores the text-graph and
graph-latent runs WITHOUT re-running the pipeline.

vLLM (Qwen3-8B) must be serving on :9999 -- model_prompting routes openai/Qwen* there.

Examples (run from repo root):
  python -m scripts.coding.eval_coding_solutions \
      --solution-dir marble/logs/qwen3-8b/coding_solutions        --tag text
  python -m scripts.coding.eval_coding_solutions \
      --solution-dir marble/logs/qwen3-8b/coding_latent_steps10   --tag latent10
"""
import argparse
import json
import os

from ruamel.yaml import YAML

from marble.evaluator.evaluator import Evaluator

CRITERIA = ["instruction_following", "executability", "consistency", "quality"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-dir", default="marble/configs/coding_configs")
    ap.add_argument(
        "--solution-dir",
        required=True,
        help="dir holding solution_<id>.py (e.g. marble/logs/qwen3-8b/coding_solutions)",
    )
    ap.add_argument("--n", type=int, default=100, help="max task id to scan")
    ap.add_argument("--model", default="openai/Qwen3-8B", help="judge LLM (vLLM)")
    ap.add_argument("--tag", default="run", help="label for output file + logs")
    ap.add_argument("--out", default=None, help="output jsonl (default result/coding_quality_<tag>.jsonl)")
    args = ap.parse_args()

    out = args.out or f"result/coding_quality_{args.tag}.jsonl"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    yaml = YAML()
    evaluator = Evaluator({"evaluate_llm": {"model": args.model}})

    totals = {k: 0.0 for k in CRITERIA}
    scored = 0
    with open(out, "w", encoding="utf-8") as fout:
        for i in range(1, args.n + 1):
            cfg_path = os.path.join(args.config_dir, f"config_{i}.yaml")
            sol_path = os.path.join(args.solution_dir, f"solution_{i}.py")
            if not (os.path.exists(cfg_path) and os.path.exists(sol_path)):
                continue
            with open(cfg_path, encoding="utf-8") as f:
                task = yaml.load(f)["task"]["content"]
            with open(sol_path, encoding="utf-8") as f:
                code = f.read()

            evaluator.evaluate_code_quality(task=task, code_result=code)
            scores = dict(evaluator.metrics.get("code_quality", {}))
            avg = round(sum(scores.get(k, 0) for k in CRITERIA) / len(CRITERIA), 3)
            fout.write(json.dumps({"id": i, "scores": scores, "avg": avg}) + "\n")
            fout.flush()
            for k in CRITERIA:
                totals[k] += scores.get(k, 0)
            scored += 1
            print(f"[{args.tag}] id={i} {scores} avg={avg}")

    if scored:
        print(f"\n=== {args.tag}: {scored} solutions scored ===")
        for k in CRITERIA:
            print(f"  {k:22s}: {totals[k] / scored:.3f}")
        print(f"  {'OVERALL':22s}: {sum(totals.values()) / (len(CRITERIA) * scored):.3f}")
    else:
        print("No (config, solution) pairs found -- check --solution-dir / --config-dir.")
    print(f"per-task scores -> {out}")


if __name__ == "__main__":
    main()
