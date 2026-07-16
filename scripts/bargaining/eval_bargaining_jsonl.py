#!/usr/bin/env python3
"""Offline buyer/seller evaluation for saved MARBLE bargaining runs.

This automates the original MARBLE procedure documented in
``scripts/world/readme.md``: evaluate the same final-round summary twice, once
with the original buyer prompt and once with the original seller prompt.  The
scoring protocol is intentionally unchanged:

* input = ``row["task"]`` and ``row["iterations"][-1]["summary"]``;
* one user message per role;
* ``max_tokens=512`` by default, ``temperature=0.0``, ``top_p=None``;
* the original ``Evaluator.parse_task_world_evaluation`` parser, including its
  greedy JSON regex and ``-1`` sentinel on malformed output;
* no JSON repair, format retry, prompt rewrite, or score clamping.

The scored output copies every source row and replaces only
``task_evaluation``.  Raw judge responses go to a separate sidecar JSONL so the
main file remains compatible with ``scripts/summarize_results.py``.

Examples (an OpenAI-compatible local server must already be running)::

    python -m scripts.bargaining.eval_bargaining_jsonl \
      result/bargaining_graph_solutions.jsonl \
      --out result/bargaining_graph_offline_eval.jsonl \
      --model openai/Qwen3-8B \
      --base-url http://localhost:9999/v1 \
      --max-tokens 2048

    python -m scripts.bargaining.eval_bargaining_jsonl \
      result/bargaining_graph_latent_steps10_memory512_plan.jsonl \
      --out result/bargaining_latent_offline_eval.jsonl \
      --model openai/My-Other-Local-Model \
      --base-url http://localhost:8000/v1 \
      --resume

The local model may differ from the paper's judge.  In that case this reproduces
the original evaluation *protocol*, not the paper's absolute scores.
"""

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

import litellm

from marble.evaluator.evaluator import Evaluator


ROLES: Tuple[str, str] = ("buyer", "seller")
CRITERIA: Tuple[str, str, str] = (
    "effectiveness_of_strategies",
    "progress_and_outcome",
    "interaction_dynamics",
)


def load_jsonl(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    """Load non-empty JSONL rows while preserving their physical line number."""
    rows: List[Tuple[int, Dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append((line_number, row))
    return rows


def extract_original_eval_input(
    row: Dict[str, Any], source: Path, line_number: int
) -> Tuple[str, str]:
    """Return exactly the task and final summary used by graph_coordinate()."""
    task = row.get("task")
    iterations = row.get("iterations")
    if not isinstance(task, str):
        raise ValueError(f"{source}:{line_number}: missing string field 'task'")
    if not isinstance(iterations, list) or not iterations:
        raise ValueError(f"{source}:{line_number}: missing non-empty 'iterations'")
    final_iteration = iterations[-1]
    if not isinstance(final_iteration, dict):
        raise ValueError(
            f"{source}:{line_number}: final iteration must be a JSON object"
        )
    summary = final_iteration.get("summary")
    if not isinstance(summary, str):
        raise ValueError(
            f"{source}:{line_number}: missing string 'iterations[-1].summary'"
        )
    return task, summary


def qwen_extra_body(model: str) -> Dict[str, Any] | None:
    """Mirror this checkout's local-Qwen behavior without affecting other models."""
    if "qwen3" in model.lower():
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def call_local_judge(
    *, model: str, base_url: str, api_key: str, prompt: str, max_tokens: int
) -> str:
    """Make one original-parameter LLM-as-a-judge call."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "n": 1,
        "temperature": 0.0,
        "top_p": None,
        "stream": None,
        "base_url": base_url,
        "api_key": api_key,
    }
    extra_body = qwen_extra_body(model)
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    completion = litellm.completion(**kwargs)
    content = completion.choices[0].message.content
    return content if isinstance(content, str) else ""


def evaluate_role(
    evaluator: Evaluator,
    *,
    role: str,
    task: str,
    result: str,
    model: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
) -> Tuple[Dict[str, int], str]:
    """Run one unchanged original role prompt and return that role's parsed scores."""
    prompt_template = evaluator.evaluation_prompts["world"]["task_evaluation"][
        f"{role}_prompt"
    ]
    prompt = prompt_template.format(task=task, result=result)
    raw_response = call_local_judge(
        model=model,
        base_url=base_url,
        api_key=api_key,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    parsed = evaluator.parse_task_world_evaluation(raw_response)
    # Each original role prompt returns only its own top-level key.  Keep only
    # that role so the default -1 values for the opposite role cannot overwrite
    # the result from the other independent call.
    return dict(parsed[role]), raw_response


def role_is_valid(scores: Dict[str, Any]) -> bool:
    """The original parser's only missing-value signal is -1."""
    return all(scores.get(criterion, -1) != -1 for criterion in CRITERIA)


def default_raw_path(out_path: Path) -> Path:
    if out_path.suffix:
        return out_path.with_name(f"{out_path.stem}.raw{out_path.suffix}")
    return out_path.with_name(f"{out_path.name}.raw.jsonl")


def report_scores(
    scored_rows: Sequence[Dict[str, Any]], *, source_total: int
) -> None:
    """Report role coverage and the paper-style score only when fully valid."""
    print(f"\nScored output coverage: {len(scored_rows)}/{source_total} task(s)")
    role_means: Dict[str, float] = {}
    role_valid_counts: Dict[str, int] = {}

    for role in ROLES:
        valid: List[Dict[str, Any]] = []
        for row in scored_rows:
            task_eval = row.get("task_evaluation", {})
            scores = task_eval.get(role, {}) if isinstance(task_eval, dict) else {}
            if isinstance(scores, dict) and role_is_valid(scores):
                valid.append(scores)

        role_valid_counts[role] = len(valid)
        if valid:
            role_mean = mean(
                float(scores[criterion])
                for scores in valid
                for criterion in CRITERIA
            )
            role_means[role] = role_mean
            completeness = "complete" if len(valid) == source_total else "partial"
            print(
                f"{role:6s}: valid={len(valid)}/{source_total}, "
                f"{completeness}_mean={role_mean:.3f}"
            )
        else:
            print(f"{role:6s}: valid=0/{source_total}, no mean")

    full_coverage = len(scored_rows) == source_total
    full_validity = all(
        role_valid_counts.get(role, 0) == source_total for role in ROLES
    )
    if full_coverage and full_validity and source_total:
        bargaining_score = mean(role_means[role] for role in ROLES)
        print(f"Bargaining score (1-5): {bargaining_score:.3f}")
        print(f"Bargaining Task Score (%): {bargaining_score * 20:.2f}")
    else:
        print(
            "Paper-style Bargaining score unavailable: every source task must "
            "have valid buyer and seller ratings. Partial means above are "
            "diagnostics only."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, help="graph/graph-latent result JSONL")
    parser.add_argument("--out", type=Path, required=True, help="scored output JSONL")
    parser.add_argument(
        "--raw-out",
        type=Path,
        default=None,
        help="raw judge sidecar (default: <out-stem>.raw.jsonl)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("BARGAINING_JUDGE_MODEL", "openai/Qwen3-8B"),
        help="LiteLLM model name served by the local OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_API_BASE", "http://localhost:9999/v1"),
        help="local OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="local API key (default: OPENAI_API_KEY or EMPTY)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="maximum judge output tokens (original MARBLE default: 512)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="score at most this many new rows (useful for a smoke test)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append after already completed rows in --out",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    out_path = args.out.resolve()
    raw_path = (args.raw_out or default_raw_path(args.out)).resolve()

    if input_path == out_path:
        raise ValueError("--out must differ from the input JSONL")
    if raw_path in (input_path, out_path):
        raise ValueError("--raw-out must differ from both input and --out")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")

    source_rows = load_jsonl(input_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    completed_rows: List[Tuple[int, Dict[str, Any]]] = []
    if args.resume and out_path.exists():
        completed_rows = load_jsonl(out_path)
    completed = len(completed_rows)
    if completed > len(source_rows):
        raise ValueError(
            f"--out contains {completed} rows but input contains only "
            f"{len(source_rows)}"
        )

    pending = source_rows[completed:]
    if args.limit is not None:
        pending = pending[: args.limit]

    evaluator = Evaluator({"evaluate_llm": {"model": args.model}})
    mode = "a" if args.resume else "w"
    raw_mode = "a" if args.resume else "w"

    with out_path.open(mode, encoding="utf-8") as fout, raw_path.open(
        raw_mode, encoding="utf-8"
    ) as fraw:
        for offset, (source_line, row) in enumerate(pending, completed + 1):
            task, result = extract_original_eval_input(row, input_path, source_line)
            ratings: Dict[str, Dict[str, int]] = {}
            raw_responses: Dict[str, str] = {}
            parse_ok: Dict[str, bool] = {}

            for role in ROLES:
                role_scores, raw_response = evaluate_role(
                    evaluator,
                    role=role,
                    task=task,
                    result=result,
                    model=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    max_tokens=args.max_tokens,
                )
                ratings[role] = role_scores
                raw_responses[role] = raw_response
                parse_ok[role] = role_is_valid(role_scores)

            scored_row = dict(row)
            scored_row["task_evaluation"] = ratings
            fout.write(json.dumps(scored_row, ensure_ascii=False) + "\n")
            fout.flush()

            raw_record = {
                "source_path": str(input_path),
                "source_line": source_line,
                "model": args.model,
                "base_url": args.base_url,
                "judge_parameters": {
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                    "top_p": None,
                },
                "raw_response": raw_responses,
                "parsed": ratings,
                "parse_ok": parse_ok,
            }
            fraw.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
            fraw.flush()
            status = " ".join(
                f"{role}={'ok' if parse_ok[role] else 'MISS'}" for role in ROLES
            )
            print(f"[{offset}/{len(source_rows)}] source_line={source_line} {status}")

    final_rows = [row for _, row in load_jsonl(out_path)] if out_path.exists() else []
    report_scores(final_rows, source_total=len(source_rows))
    print(f"Scored rows: {out_path}")
    print(f"Raw judge responses: {raw_path}")


if __name__ == "__main__":
    main()
