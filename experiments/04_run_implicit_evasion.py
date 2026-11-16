#!/usr/bin/env python3
"""Phase 4: Implicit Non-Completion & Plausible-Failure Evasion Audit.

Evaluates n = 200 chosen vs. rejected implicit failure / non-completion pairs
from Adaption's generated dataset across Native and Hardened scorers,
reporting aggregate pass rates, separation deltas, and 95% Wilson CIs.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "benchmark-designer" / "scripts"))

from judge_meta_eval import wilson  # type: ignore  # noqa: E402
from targets.darkbench_native import score_darkbench_native
from targets.syceval_native import score_syceval_native
from verifier.score import score_darkbench_hardened, score_syceval_hardened


def load_pairs(n: int) -> tuple[str, list[dict[str, Any]]]:
    pref_path = ROOT / "data" / "implicit_non_completion_pref.jsonl"
    fallback_path = ROOT / "data" / "implicit_llm_failure_pairs.jsonl"

    # Check if implicit_non_completion_pref finished on Adaption
    if not pref_path.exists():
        try:
            from adaption import Adaption

            c = Adaption()
            info = c.datasets.get("3bc37f63-ffd3-4525-ad30-0913ac499fe2")
            if info.status == "succeeded":
                c.datasets.download(info.dataset_id, file_format="jsonl").write_to_file(
                    str(pref_path)
                )
        except Exception:  # noqa: BLE001
            pass

    path = pref_path if pref_path.exists() else fallback_path
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= n:
                    break
    return path.name, rows


def eval_pair_suite(
    suite_name: str,
    pairs: list[dict[str, Any]],
    scorer_fn: Any,
    workers: int,
) -> dict[str, Any]:
    chosen_scores: list[float] = [0.0] * len(pairs)
    rejected_scores: list[float] = [0.0] * len(pairs)

    def _run_one(idx: int, is_chosen: bool) -> tuple[int, bool, float]:
        row = pairs[idx]
        prompt = row.get("enhanced_prompt") or row.get("prompt", "")
        chosen = row["chosen"]
        rejected = row["rejected"]
        task = {
            "id": f"implicit-{idx:03d}",
            "prompt": prompt,
            "input": prompt,
            "target": chosen,
        }
        out = chosen if is_chosen else rejected
        res = scorer_fn(out, task)
        return idx, is_chosen, float(res.get("score", 0.0))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = []
        for i in range(len(pairs)):
            futs.append(pool.submit(_run_one, i, True))
            futs.append(pool.submit(_run_one, i, False))
        for fut in as_completed(futs):
            try:
                idx, is_c, sc = fut.result()
                if is_c:
                    chosen_scores[idx] = sc
                else:
                    rejected_scores[idx] = sc
            except Exception:  # noqa: BLE001
                pass

    n = len(pairs)
    c_pass = sum(1 for s in chosen_scores if s >= 1.0)
    r_pass = sum(1 for s in rejected_scores if s >= 1.0)
    c_ci = wilson(c_pass, n)
    r_ci = wilson(r_pass, n)
    return {
        "suite": suite_name,
        "n": n,
        "chosen_pass_rate": round(c_pass / n, 4),
        "chosen_wilson_95_ci": [round(c_ci[0], 4), round(c_ci[1], 4)],
        "rejected_false_pass_rate": round(r_pass / n, 4),
        "rejected_wilson_95_ci": [round(r_ci[0], 4), round(r_ci[1], 4)],
        "separation_pp": round(((c_pass - r_pass) / n) * 100, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--workers", type=int, default=24)
    args = p.parse_args()

    ds_name, pairs = load_pairs(args.n)
    print(f"Loaded {len(pairs)} pairs from data/{ds_name}")

    suites = [
        ("syceval_native", score_syceval_native),
        ("syceval_hardened", score_syceval_hardened),
        ("darkbench_native", score_darkbench_native),
        ("darkbench_hardened", score_darkbench_hardened),
    ]

    results = {"dataset": ds_name, "n_pairs": len(pairs), "suites": {}}
    for sname, fn in suites:
        res = eval_pair_suite(sname, pairs, fn, args.workers)
        results["suites"][sname] = res
        print(
            f"  {sname:<20} | chosen={res['chosen_pass_rate']:.1%} {res['chosen_wilson_95_ci']} | "
            f"rejected={res['rejected_false_pass_rate']:.1%} {res['rejected_wilson_95_ci']} | "
            f"sep={res['separation_pp']:+.1f} pp"
        )

    out_path = ROOT / "results" / "implicit_evasion_rates.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

