#!/usr/bin/env python3
"""Regenerate the voided `darkbench / glm-53-flash` cell from 03 attempt 2.

All 40 glm generation calls failed in a transient API window during the run;
the runner's blanket `except Exception` stored empty strings and the scorers
graded them (native 100% / hardened 0% — both artifacts). See CONTEXT.md.

This script reuses 03's own functions verbatim (import by path, no edits to
the runner), regenerates ONLY this config, overwrites the details JSONL, and
patches the matching cell in results/anchor_calibration.json. Every generation
error is captured per item (no silent swallowing) — if any call fails, the
script reports it and still persists what succeeded.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("run_03", ROOT / "experiments" / "03_run_anchors.py")
run_03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_03)

from judge_meta_eval import wilson  # type: ignore
from targets.darkbench_native import score_darkbench_native
from verifier.score import score_darkbench_hardened

MODEL = "glm-53-flash"
BENCH = "darkbench"


def main() -> int:
    tasks = run_03.load_tasks(40)[BENCH]
    workers = 8  # lower concurrency to sidestep provider throttling
    errors: dict[int, str] = {}
    responses: list[str] = [""] * len(tasks)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(run_03.generate_model_response, MODEL, BENCH, t): i
            for i, t in enumerate(tasks)
        }
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                responses[idx] = fut.result()
            except Exception as exc:  # noqa: BLE001
                errors[idx] = f"{type(exc).__name__}: {exc}"
                print(f"GEN ERROR item {idx}: {errors[idx][:200]}")

    n_ok = sum(1 for r in responses if r.strip())
    print(f"generation: {n_ok}/{len(tasks)} non-empty responses, {len(errors)} errors")
    if errors:
        print("NOTE: generation errors present — cell will be marked partial, not written as clean.")

    nat_scores: dict[str, float] = {}
    hard_scores: dict[str, float] = {}
    for i, (resp, task) in enumerate(zip(responses, tasks)):
        try:
            nat_scores[str(i)] = float(score_darkbench_native(resp, task).get("score", 0.0))
        except Exception as exc:  # noqa: BLE001
            print(f"NATIVE SCORE ERROR item {i}: {exc}")
            nat_scores[str(i)] = 0.0
        try:
            hard_scores[str(i)] = float(score_darkbench_hardened(resp, task).get("score", 0.0))
        except Exception as exc:  # noqa: BLE001
            print(f"HARDENED SCORE ERROR item {i}: {exc}")
            hard_scores[str(i)] = 0.0

    n = len(tasks)
    nat_s = sum(1 for v in nat_scores.values() if v >= 1.0)
    hard_s = sum(1 for v in hard_scores.values() if v >= 1.0)
    nat_ci = wilson(nat_s, n)
    hard_ci = wilson(hard_s, n)

    # Evidence layer (same schema as 03)
    details_path = ROOT / "results" / "anchor_details" / f"{BENCH}__{MODEL}.jsonl"
    with open(details_path, "w", encoding="utf-8") as f:
        for i, task in enumerate(tasks):
            f.write(json.dumps({
                "index": i,
                "task_id": task.get("id") or task.get("task_id"),
                "dark_pattern": (task.get("metadata") or {}).get("dark_pattern"),
                "task_input": (task.get("input") or "")[:300],
                "response": responses[i],
                "native": nat_scores.get(str(i)),
                "hardened": hard_scores.get(str(i)),
                "regen": True,
                "gen_error": errors.get(i),
            }) + "\n")

    if errors:
        print("ABSTAINING from patching anchor_calibration.json — errors present; rerun this script.")
        return 1

    cell = {
        "config": MODEL,
        "benchmark": BENCH,
        "n": n,
        "native_pass_rate": round(nat_s / n, 4),
        "native_wilson_95_ci": [round(nat_ci[0], 4), round(nat_ci[1], 4)],
        "hardened_pass_rate": round(hard_s / n, 4),
        "hardened_wilson_95_ci": [round(hard_ci[0], 4), round(hard_ci[1], 4)],
        "delta_pp": round(((hard_s - nat_s) / n) * 100, 2),
    }
    cal_path = ROOT / "results" / "anchor_calibration.json"
    cal = json.loads(cal_path.read_text())
    cal[BENCH][MODEL] = cell
    cal_path.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(
        f"PATCHED {BENCH}/{MODEL}: native={cell['native_pass_rate']:.1%} | "
        f"hardened={cell['hardened_pass_rate']:.1%} (delta={cell['delta_pp']:+.1f} pp)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
