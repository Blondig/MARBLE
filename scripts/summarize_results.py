#!/usr/bin/env python3
"""Aggregate the per-task metrics in result/*.jsonl and compare runs side by side.

Each line of a result JSONL is one task's ``summary_data`` (see
``Engine._write_to_jsonl``). This walks the numeric metrics, flattening per-round
lists (e.g. ``planning_scores``) and nested rating dicts (``task_evaluation`` /
``code_quality`` / the offline ``scores``), then prints the mean -- with the
sample count -- per metric for each file. Give it two or more files (e.g. the
text-graph baseline and the graph-latent run) and it also prints a side-by-side
table with the latent-minus-graph delta.

``-1`` and ``null`` are MARBLE's "not scored" sentinels (parse failures, or the
KPI/milestone fields that latent leaves as N/A): they are excluded from the mean
and counted under ``miss``.

Lightweight: stdlib only, no model calls. Run from the repo root, e.g.::

  python scripts/summarize_results.py \
      result/development_output.jsonl \
      result/coding_graph_latent_steps10.jsonl

  # explicit labels (label=path):
  python scripts/summarize_results.py \
      graph=result/development_output.jsonl \
      latent=result/coding_graph_latent_steps10.jsonl

  # the offline code-quality files work too:
  python scripts/summarize_results.py \
      result/coding_quality_text.jsonl result/coding_quality_latent10.jsonl
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from statistics import mean

# Bookkeeping / free-text fields that are not metrics -- never recurse into them.
# `iterations` holds the per-round detail whose scores are already surfaced at the
# top level (planning_scores/communication_scores), so skipping it avoids double
# counting. `agent_kpis` keys are per-task agent ids, not comparable across runs.
SKIP_KEYS = {
    "task", "coordination_mode", "communication_mode", "model_name",
    "iterations", "final_output", "predicted", "root_cause", "id", "iteration",
    "continue_simulation", "agent_kpis", "task_assignments", "task_results",
    "summary", "communications",
}


def is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def collect(obj: object, name: str, out: "defaultdict[str, list]") -> None:
    """Recursively gather numeric leaves into ``out[name]``.

    Lists are flattened under the same metric name, so a per-round list like
    ``planning_scores`` contributes one sample per round.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP_KEYS or k.endswith("_note"):
                continue
            collect(v, f"{name}.{k}" if name else k, out)
    elif isinstance(obj, list):
        for v in obj:
            collect(v, name, out)
    elif is_number(obj):
        out[name].append(obj)


def label_for(path: str, records: list) -> str:
    """Derive a column label: the run's coordination_mode, else the file stem."""
    for rec in records:
        mode = rec.get("coordination_mode") if isinstance(rec, dict) else None
        if mode:
            return str(mode)
    return os.path.splitext(os.path.basename(path))[0]


def load(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  warn: {path}:{ln} skipped (bad json: {e})", file=sys.stderr)
    return records


def summarize(records: list) -> "dict[str, tuple[float, int, int]]":
    """metric -> (mean_of_valid, n_valid, n_missing). Valid means a number >= 0;
    negative values are the -1 sentinel."""
    raw: "defaultdict[str, list]" = defaultdict(list)
    for rec in records:
        collect(rec, "", raw)
    stats = {}
    for metric, values in raw.items():
        valid = [v for v in values if v >= 0]
        if valid:
            stats[metric] = (mean(valid), len(valid), len(values) - len(valid))
    return stats


def parse_arg(arg: str) -> "tuple[str | None, str]":
    """`label=path` -> (label, path); otherwise (None, path)."""
    if "=" in arg and not os.path.exists(arg):
        label, _, path = arg.partition("=")
        return label or None, path
    return None, arg


def fmt(x: float) -> str:
    return f"{x:,.3f}"


# --- token_breakdown helpers (engine writes summary_data["token_breakdown"]) ----
BD_PREFIX = "token_breakdown."
# Stage order for display; "total" last. communication is the one that differs
# between graph and graph-latent, so it leads.
BD_STAGES = [
    "communication",
    "memory_latent",
    "agent_reasoning",
    "env_tools",
    "planner",
    "total",
]
BD_PARTS = ["communication", "memory_latent", "agent_reasoning", "env_tools", "planner"]


def breakdown_of(stats: dict) -> "dict[str, float]":
    """Pull the token_breakdown.* means out of a file's stats (or {} if absent)."""
    bd = {}
    for stage in BD_STAGES:
        key = BD_PREFIX + stage
        if key in stats:
            bd[stage] = stats[key][0]
    return bd


def bd_total(bd: "dict[str, float]") -> float:
    """Total for percentages: the explicit 'total', else the sum of the parts."""
    return bd.get("total") or sum(bd.get(p, 0.0) for p in BD_PARTS)


def bd_cell(bd: "dict[str, float]", stage: str) -> str:
    """`70,196 (20.3%)` for a stage, or `-` if this run has no breakdown."""
    if stage not in bd:
        return "-"
    tot = bd_total(bd) or 1.0
    return f"{bd[stage]:,.0f} ({100 * bd[stage] / tot:.1f}%)"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="+", help="result JSONL file(s), optionally label=path")
    args = ap.parse_args()

    columns = []  # (label, stats)
    for arg in args.files:
        label, path = parse_arg(arg)
        if not os.path.exists(path):
            print(f"warn: no such file: {path}", file=sys.stderr)
            continue
        records = load(path)
        label = label or label_for(path, records)
        stats = summarize(records)
        columns.append((label, stats))

        print(f"\n=== {label}  ({path}, {len(records)} task(s)) ===")
        if not stats:
            print("  (no numeric metrics found)")
            continue
        # token_breakdown.* is shown in its own % block below, not as flat rows.
        generic = {m: v for m, v in stats.items() if not m.startswith(BD_PREFIX)}
        if generic:
            width = max(len(m) for m in generic)
            for metric in sorted(generic):
                m, n, miss = generic[metric]
                miss_s = f", miss={miss}" if miss else ""
                print(f"  {metric:<{width}}  mean={fmt(m)}  (n={n}{miss_s})")
        bd = breakdown_of(stats)
        if bd:
            tot = bd_total(bd) or 1.0
            print("  token breakdown (mean tokens | % of total):")
            sw = max(len(s) for s in bd)
            for stage in BD_STAGES:
                if stage in bd:
                    print(f"    {stage:<{sw}}  {bd[stage]:>14,.0f}  {100 * bd[stage] / tot:5.1f}%")

    if len(columns) < 2:
        return

    # Side-by-side comparison over the union of metrics (breakdown shown below).
    print("\n=== comparison (mean) ===")
    all_metrics = sorted(
        {m for _, s in columns for m in s if not m.startswith(BD_PREFIX)}
    )
    labels = [lbl for lbl, _ in columns]
    mwidth = max([len(m) for m in all_metrics] + [6])
    cwidth = max([len(l) for l in labels] + [9])
    header = f"  {'metric':<{mwidth}}  " + "  ".join(f"{l:>{cwidth}}" for l in labels)
    if len(columns) == 2:
        header += f"  {'delta':>{cwidth}}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for metric in all_metrics:
        cells = []
        vals = []
        for _, stats in columns:
            if metric in stats:
                v = stats[metric][0]
                vals.append(v)
                cells.append(f"{fmt(v):>{cwidth}}")
            else:
                vals.append(None)
                cells.append(f"{'-':>{cwidth}}")
        row = f"  {metric:<{mwidth}}  " + "  ".join(cells)
        if len(columns) == 2 and None not in vals:
            row += f"  {vals[1] - vals[0]:>+{cwidth},.3f}"
        print(row)
    if len(columns) == 2:
        print(f"\n  (delta = {labels[1]} - {labels[0]})")

    # Dedicated token-breakdown comparison (mean tokens | % of total per column).
    bds = [(lbl, breakdown_of(s)) for lbl, s in columns]
    if any(bd for _, bd in bds):
        print("\n=== token breakdown (mean tokens | % of total) ===")
        cells_by_stage = {
            st: [bd_cell(bd, st) for _, bd in bds] for st in BD_STAGES
        }
        swidth = max(len(s) for s in BD_STAGES)
        cw = max([len(c) for cs in cells_by_stage.values() for c in cs] + [len(l) for l in labels])
        print(f"  {'stage':<{swidth}}  " + "  ".join(f"{l:>{cw}}" for l in labels))
        print("  " + "-" * (swidth + 2 + (cw + 2) * len(labels)))
        for stage in BD_STAGES:
            cells = "  ".join(f"{c:>{cw}}" for c in cells_by_stage[stage])
            print(f"  {stage:<{swidth}}  {cells}")
        missing = [lbl for lbl, bd in bds if not bd]
        if missing:
            print(
                f"\n  note: no token_breakdown in: {', '.join(missing)} "
                "(re-run with the current engine to get it)."
            )


if __name__ == "__main__":
    main()
