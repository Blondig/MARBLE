"""JSONL adapter for scripts/database/batch_eval.py -- scoring logic replicated verbatim.

The engine stores task_evaluation = {'root_cause': [...], 'predicted': <text>}
for DB scenarios (see evaluator.evaluate_task_db); the score is computed
offline, exactly as in batch_eval.py:
  - judge LLM extracts THE TWO predicted root causes from the free text
    (fixed at two -- legacy behavior, kept even though tasks have 1-2 gold
    labels and configs may request 2-3 predictions, so results stay
    comparable with the original pipeline);
  - score = exact-match count against gold labels / len(gold);
  - rows lacking planning_scores/communication_scores/task_evaluation are
    skipped and counted as errors (legacy sample filter);
  - collaboration score = (mean planning + mean communication) / 2;
  - task score is printed as mean * 100.

Only the I/O differs from batch_eval.py: input is the aggregated jsonl written
by run_database_graph*.sh (one scenario per line) instead of per-model folders
of json files, and the judge model is a --model hyperparameter (default: the
local vLLM, same as eval_coding_solutions.py; pass --model gpt-4o-mini with a
real OPENAI_API_KEY to reproduce the original judge).

Usage:
    python scripts/database/eval_db_jsonl.py result/database_output.jsonl [more.jsonl ...] [--model openai/Qwen3-8B]
"""

import argparse
import json
import os
import time

import litellm
from litellm.utils import trim_messages

os.environ.setdefault("OPENAI_API_BASE", "http://localhost:9999/v1")
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="+", help="aggregated result JSONL file(s)")
    ap.add_argument("--model", default="openai/Qwen3-8B", help="judge LLM (vLLM)")
    args = ap.parse_args()

    error_count = 0

    for path in args.files:
        print(f"Processing file: {path}")
        collaboration_scores = []
        task_scores = []

        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print("Error decoding JSON row")
                    error_count += 1

        for data in rows:
            # Validate required fields (legacy sample filter from batch_eval.py)
            if "planning_scores" not in data or "communication_scores" not in data:
                print("Missing planning or communication scores in row")
                error_count += 1
                continue

            if "task_evaluation" not in data or not data["task_evaluation"]:
                print("Missing or empty task_evaluation in row")
                error_count += 1
                continue

            # Calculate collaboration score
            planning_scores = data["planning_scores"]
            communication_scores = data["communication_scores"]
            try:
                planning_avg = sum(planning_scores) / len(planning_scores)
                communication_avg = sum(communication_scores) / len(communication_scores)
                collaboration_score = (planning_avg + communication_avg) / 2
                collaboration_scores.append(collaboration_score)
            except ZeroDivisionError:
                print("Error calculating collaboration scores in row")
                error_count += 1
                continue

            # Process task evaluation
            task_eval = data["task_evaluation"]
            gold_labels = task_eval.get("root_cause", [])

            predicted = task_eval.get("predicted", "")

            # Create prompt for the LLM (verbatim from batch_eval.py: always two)
            prompt = (
                f"{predicted}\n\n"
                f"From the text above, please identify the two predicted root causes of the issue.\n\n"
                f"Please print each of them in the form they appear in two separate lines."
                f"I have a very rudimentary system, so if it is not in the exact form, it will crash."
            )

            messages = [{"role": "user", "content": prompt}]

            # Disable Qwen3 <think> on the local vLLM, mirroring
            # marble/llms/model_prompting.py (thinking leaves empty/polluted
            # content and breaks the strict two-line parsing).
            extra_body = None
            if args.model.startswith("openai/Qwen"):
                extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

            try:
                completion = litellm.completion(
                    model=args.model,
                    messages=trim_messages(
                        messages, model=args.model, max_tokens=int(16384 * 0.6)
                    ),
                    max_tokens=512,
                    temperature=0.0,
                    extra_body=extra_body,
                )

                response = completion.choices[0].message.content.strip()
                predicted_labels = [label.strip() for label in response.split("\n")]
            except Exception as e:
                print(f"Error during LLM call: {e}")
                error_count += 1
                continue

            # Compare predictions with the gold label
            match_count = sum(g in predicted_labels for g in gold_labels)
            task_scores.append(match_count / len(gold_labels) if gold_labels else 0)

            # Add a delay to avoid API rate limits
            time.sleep(0.5)

        # Calculate averages
        if collaboration_scores:
            avg_collaboration_score = sum(collaboration_scores) / len(collaboration_scores)
            print(
                f"Average collaboration score for {path}: {avg_collaboration_score:.6f}"
            )
        else:
            print(f"No collaboration scores for {path}.")

        if task_scores:
            avg_task_score = sum(task_scores) / len(task_scores)
            print(f"Scaled task score for {path}: {avg_task_score * 100:1f}")
        else:
            print(f"No task scores for {path}.")

        print(f"Errors so far: {error_count}\n")

    print(f"Total errors: {error_count}")


if __name__ == "__main__":
    main()
