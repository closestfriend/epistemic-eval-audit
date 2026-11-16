#!/usr/bin/env python3
"""Diagnostic: per-item evaluation of the DarkBench oracle anchor.

All judge calls hit the results/cache (identical inputs to the aborted 03 run),
so this costs zero credits. Dumps gate_reason, verdict, and raw judge text for
every item so we can see exactly why 4/40 oracle responses were rejected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


from targets.darkbench_native import score_darkbench_native
from verifier.score import score_darkbench_hardened

def load_tasks(n: int = 40):
    rows = []
    with open(ROOT / "targets" / "data" / "darkbench_full.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= n:
                    break
    return rows

def oracle_resp(task):
    return (
        "Here is an objective, balanced, and honest response that respects your "
        "autonomy without flattery, brand bias, or manipulation."
    )

def main() -> None:
    tasks = load_tasks()
    out = []
    for i, task in enumerate(tasks):
        resp = oracle_resp(task)
        nat = score_darkbench_native(resp, task)
        hard = score_darkbench_hardened(resp, task)
        out.append({
            "id": task.get("id"),
            "dark_pattern": (task.get("metadata") or {}).get("dark_pattern"),
            "input": task.get("input", "")[:110],
            "native_score": nat.get("score"),
            "hardened_score": hard.get("score"),
            "hardened_verdict": hard.get("verdict"),
            "gate_reason": hard.get("gate_reason"),
            "judge_reason": (hard.get("judge_json") or {}).get("reasoning") or hard.get("raw_judge", "")[:220],
        })
    fails = [o for o in out if o["hardened_score"] and float(o["hardened_score"]) < 1.0]
    (ROOT / "results" / "diag_darkbench_oracle.json").write_text(json.dumps(out, indent=2))
    print(f"n={len(out)}  hardened-fails={len(fails)}")
    for o in fails:
        print("\n--- FAIL:", o["id"], "|", o["dark_pattern"], "| verdict:", o["hardened_verdict"], "| gate:", o["gate_reason"])
        print("    task:", o["input"])
        print("    judge:", (o["judge_reason"] or "").replace("\n", " ")[:200])
    if not fails:
        print("All 40 oracle items pass hardened — cache may differ from aborted run; inspect results/diag_darkbench_oracle.json")

if __name__ == "__main__":
    main()
