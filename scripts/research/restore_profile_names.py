#!/usr/bin/env python3
"""Generate research configs with REAL researcher names restored.

The upstream pipeline that built ``multiagentbench/research/research_main.jsonl``
dropped each researcher's real ``name`` (e.g. "Yu Zhou") and kept only the
anonymous ``bio`` under ``agent_id: agent1..N``. The research env's author-lookup
tool (``collect_publications_and_coauthors``) needs a real name, so a local model
just passes "agent1" -> Semantic Scholar finds nothing -> 429 / retries.

The names still live in ``data/research-data/profile_dbs/profile_*/ProfileDB.json``
(keyed by uuid, with a matching ``bio``). This script:

  1. builds a ``bio -> name`` index from those ProfileDBs;
  2. reads each task in research_main.jsonl, matches every agent by exact ``bio``
     and prepends ``"My name is <name>. "`` to its ``profile``;
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


def build_index(profile_dbs_dir: str) -> "dict[str, str]":
    """bio (stripped) -> name, from every profile_*/ProfileDB.json."""
    idx: "dict[str, str]" = {}
    pattern = os.path.join(profile_dbs_dir, "profile_*", "ProfileDB.json")
    files = glob.glob(pattern)
    if not files:
        sys.exit(f"ERROR: no ProfileDB.json under {pattern}")
    for path in files:
        with open(path, encoding="utf-8") as f:
            db = json.load(f)
        for entry in db.values():
            bio = (entry.get("bio") or "").strip()
            name = (entry.get("name") or "").strip()
            if bio and name:
                idx[bio] = name
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--research-jsonl", default="multiagentbench/research/research_main.jsonl")
    ap.add_argument("--profile-dbs", default="data/research-data/profile_dbs")
    ap.add_argument("--out-dir", default="multiagentbench/research_yaml_files_named")
    ap.add_argument("--verify-only", action="store_true", help="report coverage, do not write")
    args = ap.parse_args()

    idx = build_index(args.profile_dbs)
    print(f"indexed {len(idx)} bio->name entries from {args.profile_dbs}")

    if not args.verify_only:
        os.makedirs(args.out_dir, exist_ok=True)

    n_tasks = n_agents = n_named = 0
    missing = []  # (task_id, agent_id)
    with open(args.research_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = fill_defaults(json.loads(line))
            task_id = data.get("task_id", n_tasks + 1)
            n_tasks += 1
            for agent in data.get("agents", []):
                if not isinstance(agent, dict) or "profile" not in agent:
                    continue
                n_agents += 1
                profile = (agent["profile"] or "").strip()
                if profile.startswith(PREFIX):
                    n_named += 1
                    continue
                name = idx.get(profile)
                if name:
                    agent["profile"] = f"{PREFIX}{name}. {profile}"
                    n_named += 1
                else:
                    missing.append((task_id, agent.get("agent_id")))
            if not args.verify_only:
                out_path = os.path.join(args.out_dir, f"task_{task_id}.yaml")
                with open(out_path, "w", encoding="utf-8") as out_f:
                    yaml.dump(data, out_f, allow_unicode=True, sort_keys=False)

    dest = "(verify-only, nothing written)" if args.verify_only else args.out_dir
    print(f"tasks={n_tasks}  agents={n_agents}  named={n_named}  missing={len(missing)}  -> {dest}")
    if missing:
        print("FAIL: these agents had no bio->name match (do NOT run until resolved):")
        for tid, aid in missing[:20]:
            print(f"  task {tid}  {aid}")
        sys.exit(1)
    print("OK: every agent in every task got a real name.")


if __name__ == "__main__":
    main()
