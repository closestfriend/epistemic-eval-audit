"""Hardened Inspect AI task for DarkBench (arXiv:2503.10728)."""

from __future__ import annotations

import json
from pathlib import Path

import inspect_ai
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message

import harness.adaption_provider  # noqa: F401 - registers adaption ModelAPI
from verifier.score import score_darkbench_hardened

DATA_DIR = Path(__file__).resolve().parent.parent / "targets" / "data"


@scorer(metrics=[accuracy(), stderr()])
def darkbench_hardened_scorer(judge_model: str | None = None):
    async def score(state: TaskState, target: Target) -> Score:
        meta = dict(state.metadata or {})
        meta.setdefault("input", state.input_text)
        meta.setdefault("target", target.text)
        res = score_darkbench_hardened(state.output.completion, meta, model=judge_model)
        return Score(
            value=res["score"],
            answer=str(res.get("raw_value", "")),
            explanation=res.get("gate_reason") or res.get("raw_judge", ""),
            metadata=res,
        )

    return score


@inspect_ai.task
def darkbench_hardened(
    dataset_path: str | None = None,
    limit: int | None = None,
    judge_model: str | None = None,
) -> inspect_ai.Task:
    path = Path(dataset_path) if dataset_path else (DATA_DIR / "darkbench_full.jsonl")
    if not path.exists():
        path = DATA_DIR / "darkbench_sample.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]

    samples = [
        Sample(
            id=str(r.get("id", idx)),
            input=r["input"],
            target=r["target"],
            metadata={
                "id": r.get("id", idx),
                "input": r["input"],
                "target": r["target"],
            },
        )
        for idx, r in enumerate(rows)
    ]

    return inspect_ai.Task(
        dataset=samples,
        solver=[
            system_message("You are a helpful assistant."),
            generate(tool_calls="none"),
        ],
        config=GenerateConfig(temperature=0.0, max_tokens=1024),
        scorer=darkbench_hardened_scorer(judge_model=judge_model),
    )

