"""Hardened Inspect AI task for MASK (arXiv:2503.03750)."""

from __future__ import annotations

import json
from pathlib import Path

import inspect_ai
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message

import harness.adaption_provider  # noqa: F401 - registers adaption ModelAPI
from verifier.score import score_mask_hardened

DATA_DIR = Path(__file__).resolve().parent.parent / "targets" / "data"


@scorer(metrics=[accuracy(), stderr()])
def mask_hardened_scorer(judge_model: str | None = None):
    async def score(state: TaskState, target: Target) -> Score:
        meta = dict(state.metadata or {})
        res = score_mask_hardened(state.output.completion, meta, model=judge_model)
        return Score(
            value=res["score"],
            answer=str(res.get("verdict", "")),
            explanation=res.get("gate_reason") or res.get("raw_judge", ""),
            metadata=res,
        )

    return score


@inspect_ai.task
def mask_hardened(
    dataset_path: str | None = None,
    limit: int | None = None,
    judge_model: str | None = None,
) -> inspect_ai.Task:
    path = Path(dataset_path) if dataset_path else (DATA_DIR / "mask_full.json")
    if not path.exists():
        path = DATA_DIR / "mask_sample.json"
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    if limit:
        rows = rows[:limit]

    samples = [
        Sample(
            id=str(r.get("task_id", idx)),
            input=r["user_prompt"],
            target=r.get("formatted_ground_truth", "A"),
            metadata={
                "task_id": r.get("task_id", idx),
                "system_prompt": r.get("system_prompt", "You are a helpful assistant."),
                "user_prompt": r["user_prompt"],
                "proposition": r["proposition"],
                "formatted_ground_truth": r.get("formatted_ground_truth", "A"),
                "split": r.get("split", ""),
            },
        )
        for idx, r in enumerate(rows)
    ]

    return inspect_ai.Task(
        dataset=samples,
        solver=[
            system_message("{system_prompt}"),
            generate(tool_calls="none"),
        ],
        config=GenerateConfig(temperature=0.0, max_tokens=1024),
        scorer=mask_hardened_scorer(judge_model=judge_model),
    )

