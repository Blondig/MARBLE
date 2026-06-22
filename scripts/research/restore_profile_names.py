#!/usr/bin/env python3
"""Generate research configs with REAL researcher names restored.

The upstream pipeline that built ``multiagentbench/research/research_main.jsonl``
dropped each researcher's real ``name`` (e.g. "Yu Zhou") and kept only the
anonymous ``bio`` under ``agent_id: agent1..N``. The research env's author-lookup
tool (``collect_publications_and_coauthors``) needs a real name, so a local model
just passes "agent1" -> Semantic Scholar finds nothing -> 429 / retries.

The names still live in ``data/research-data/profile_dbs/profile_*/ProfileDB.json``
(keyed by uuid, with a matching ``bio``). This script:

  1. indexes each ProfileDB by its complete, ordered tuple of agent bios;
  2. reads each task in research_main.jsonl, uniquely matches that complete bio
     tuple to a ProfileDB, then prepends ``"My name is <name>. "`` to each
     profile using the original ProfileDB order;
  3. writes ``task_<task_id>.yaml`` to a NEW dir (default
     multiagentbench/research_yaml_files_named/), applying the same default-fill
     and dump options as multiagentbench/jsonl2yaml.py so the configs are
     identical to the validated pipeline except for the injected names;
  4. verifies every agent in every task matched a name and exits non-zero if not,
     so you can confirm 100% coverage before pointing the runner at the new dir.

Run from the repo root::

    python scripts/research/restore_profile_names.py            # generate + verify
    python scripts/research/restore_profile_names.py --verify-only   # report only
"""
import argparse
import glob
import json
import os
import sys

import yaml

PREFIX = "My name is "

# Mirrors multiagentbench/runjsonl2yaml.sh (the command used to generate the
# original research configs). Only applied to keys whose value is "" -- the
# research_main.jsonl records are already complete, so this is effectively a
# no-op, kept only to stay byte-identical to the validated pipeline.
DEFAULTS = {
    "coordinate_mode": "graph",
    "environment": {"max_iterations": 5, "name": "Research Collaboration Environment", "type": "Research"},
    "llm": "gpt-3.5-turbo",
    "memory": {"type": "BaseMemory"},
    "metrics": {"evaluate_llm": "gpt-4o"},
    "output": {"file_path": "result/discussion_output.jsonl"},
}


def fill_defaults(data: dict) -> dict:
    """Same semantics as jsonl2yaml.fill_defaults: only fill empty-string values."""
    if data.get("coordinate_mode") == "":
        data["coordinate_mode"] = DEFAULTS["coordinate_mode"]
    if data.get("llm") == "":
        data["llm"] = DEFAULTS["llm"]
    for key in ("environment", "memory", "output"):
        if isinstance(data.get(key), dict):
            for sub_key, default_val in DEFAULTS[key].items():
                if data[key].get(sub_key) == "":
                    data[key][sub_key] = default_val
    if isinstance(data.get("metrics"), dict):
        if data["metrics"].get("evaluate_llm") == "":
            data["metrics"]["evaluate_llm"] = DEFAULTS["metrics"]["evaluate_llm"]
    return data


def build_profile_groups(
    profile_dbs_dir: str,
) -> "dict[tuple[str, ...], tuple[str, ...]]":
    """Map each ProfileDB's ordered bio tuple to its ordered name tuple."""
    groups: "dict[tuple[str, ...], tuple[str, ...]]" = {}
    pattern = os.path.join(profile_dbs_dir, "profile_*", "ProfileDB.json")
    files = glob.glob(pattern)
    if not files:
        sys.exit(f"ERROR: no ProfileDB.json under {pattern}")
    for path in files:
        with open(path, encoding="utf-8") as f:
            db = json.load(f)
        entries = list(db.values())
        bios = tuple((entry.get("bio") or "").strip() for entry in entries)
        names = tuple((entry.get("name") or "").strip() for entry in entries)
        if not entries or any(not bio for bio in bios) or any(not name for name in names):
            sys.exit(f"ERROR: missing bio/name in {path}")
        if bios in groups:
            sys.exit(f"ERROR: duplicate ordered profile group in {path}")
        groups[bios] = names
    return groups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--research-jsonl", default="multiagentbench/research/research_main.jsonl")
    ap.add_argument("--profile-dbs", default="data/research-data/profile_dbs")
    ap.add_argument("--out-dir", default="multiagentbench/research_yaml_files_named")
    ap.add_argument("--verify-only", action="store_true", help="report coverage, do not write")
    args = ap.parse_args()

    groups = build_profile_groups(args.profile_dbs)
    print(f"indexed {len(groups)} ordered profile groups from {args.profile_dbs}")

    n_tasks = n_agents = n_named = 0
    generated = []  # (task_id, config)
    unmatched = []  # (task_id, agent_ids)
    seen_task_ids = set()
    with open(args.research_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = fill_defaults(json.loads(line))
            task_id = data.get("task_id", 1)
            if task_id in seen_task_ids:
                sys.exit(f"ERROR: duplicate task_id {task_id}")
            seen_task_ids.add(task_id)
            n_tasks += 1
            agents = data.get("agents", [])
            if not isinstance(agents, list) or any(
                not isinstance(agent, dict) or "profile" not in agent
                for agent in agents
            ):
                unmatched.append((task_id, []))
                continue

            bios = tuple((agent["profile"] or "").strip() for agent in agents)
            names = groups.get(bios)
            if names is None or len(names) != len(agents):
                unmatched.append(
                    (task_id, [agent.get("agent_id") for agent in agents])
                )
                continue

            for agent, bio, name in zip(agents, bios, names):
                agent["profile"] = f"{PREFIX}{name}. {bio}"
                n_agents += 1
                n_named += 1
            generated.append((task_id, data))

    dest = "(verify-only, nothing written)" if args.verify_only else args.out_dir
    print(
        f"tasks={n_tasks}  agents={n_agents}  named={n_named} "
        f"unmatched_tasks={len(unmatched)}  -> {dest}"
    )
    if unmatched:
        print("FAIL: these tasks had no unique ordered ProfileDB match:")
        for tid, agent_ids in unmatched[:20]:
            print(f"  task {tid}  agents={agent_ids}")
        sys.exit(1)

    if not args.verify_only:
        os.makedirs(args.out_dir, exist_ok=True)
        for task_id, data in generated:
            out_path = os.path.join(args.out_dir, f"task_{task_id}.yaml")
            with open(out_path, "w", encoding="utf-8") as out_f:
                yaml.dump(data, out_f, allow_unicode=True, sort_keys=False)
    print("OK: every agent in every task got a real name.")


if __name__ == "__main__":
    main()
