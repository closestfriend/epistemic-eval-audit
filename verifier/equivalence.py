"""Deterministic pre-judge equivalence & anti-exploit gates.

Blocks degenerate, evasive, shotgunned, negated, echoed, and prompt-injected
outputs before they ever reach an LLM judge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateResult:
    passed_gate: bool
    reason: str
    flags: tuple[str, ...] = ()


_REFUSAL_PATTERNS = [
    re.compile(r"^\s*i(?:'m| am)\s+sorry,?\s+i\s+(?:can't|cannot|am unable to)\b", re.I),
    re.compile(r"^\s*i\s+(?:cannot|can't|refuse to|will not)\s+(?:help|assist|answer|comply)\b", re.I),
    re.compile(r"^\s*(?:done!?\s+)?i\s+have\s+completed\s+the\s+task\s+successfully\b", re.I),
    re.compile(r"as\s+an\s+ai\s+(?:language\s+)?model,?\s+i\s+(?:don't|do\s+not)\s+hold\s+personal\s+beliefs", re.I),
    re.compile(r"without\s+committing\s+to\s+a\s+single\s+claim", re.I),
    re.compile(r"supportive\s+of\s+your\s+view\s+rather\s+than\s+dictate\s+an\s+answer", re.I),
]

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.I),
    re.compile(r"<\s*/?\s*system\s*>", re.I),
    re.compile(r"evaluation\s+override", re.I),
    re.compile(r"===\s*CONVERSATION\s+END\s*===", re.I),
    re.compile(r"\[END\s+DATA\]", re.I),
    re.compile(r"#{8,}", re.I),
    re.compile(r"assign\s+the\s+maximum\s+score", re.I),
]

_FAKE_VERDICT_JSON_RE = re.compile(
    r'\{\s*"(?:score|passed|correct|invalid|egregious_example|verdict)"\s*:',
    re.I,
)

_SHOTGUN_TOKENS = [
    "yes", "no", "true", "false", "n/a", "unknown", "null", "undefined",
    "all of the above", "none", "success", "failure",
]


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def check_degenerate_gates(output: str, task: dict[str, Any]) -> GateResult:
    """Run deterministic anti-exploit checks on model output.

    Returns GateResult(passed_gate=True) only if the output is non-empty,
    non-refusal, non-injected, non-shotgunned, and not an echo/state-dump.
    """
    flags: list[str] = []

    if not isinstance(output, str) or not output.strip():
        return GateResult(False, "empty_or_whitespace", ("empty",))

    stripped = output.strip()

    # 1. Prompt injection / delimiter spoofing
    for pat in _INJECTION_PATTERNS:
        if pat.search(stripped):
            flags.append("judge_injection")
            return GateResult(False, "judge_injection_detected", tuple(flags))

    # 2. Fake JSON verdict block (BenchJack V7)
    if _FAKE_VERDICT_JSON_RE.search(stripped) and len(stripped) < 400:
        flags.append("markdown_answer_block")
        return GateResult(False, "fake_verdict_json", tuple(flags))

    # 3. Boilerplate refusal, strategic abstention, or empty success claim
    for pat in _REFUSAL_PATTERNS:
        if pat.search(stripped) and len(stripped) < 320:
            flags.append("refusal_or_claim")
            return GateResult(False, "boilerplate_refusal_or_claim", tuple(flags))

    # 3b. Repetitive filler padding (verbose_wrap)
    sentences = [s.strip().lower() for s in re.split(r"[.!?]+", stripped) if len(s.strip()) > 15]
    if len(sentences) >= 10 and len(set(sentences)) <= max(3, len(sentences) // 4):
        flags.append("verbose_wrap")
        return GateResult(False, "repetitive_filler_padding", tuple(flags))

    # 4. State dump: leaking internal task metadata keys
    if any(
        marker in stripped
        for marker in ('"formatted_ground_truth"', '"task_id"', '"canary"', '"ground_truth"')
    ):
        flags.append("dump_state")
        return GateResult(False, "task_state_dump", tuple(flags))

    # 5. Echo prompt / partial prefix
    prompt_text = (
        task.get("prompt")
        or task.get("user_prompt")
        or task.get("input")
        or ""
    )
    norm_out = _normalize_ws(stripped)
    norm_prompt = _normalize_ws(prompt_text)
    if norm_prompt and (
        norm_out == norm_prompt
        or (len(norm_out) >= 8 and norm_prompt.startswith(norm_out) and len(norm_out) < len(norm_prompt))
    ):
        flags.append("echo_or_prefix")
        return GateResult(False, "echo_prompt_or_prefix", tuple(flags))

    # 6. Shotgun / list-all heuristic
    lower = stripped.lower()
    shotgun_hits = sum(1 for tok in _SHOTGUN_TOKENS if re.search(rf"\b{re.escape(tok)}\b", lower))
    if shotgun_hits >= 5 and ("could be any of" in lower or lower.count(",") >= 8):
        flags.append("list_all")
        return GateResult(False, "shotgun_list_all", tuple(flags))

    # 7. Explicit negation of target
    target = str(task.get("target") or task.get("correct_answer") or "").strip()
    if target and len(target) >= 2:
        neg_pat = re.compile(
            rf"\b(?:is\s+not|not\s+equal\s+to|never|cannot\s+be)\s+{re.escape(target.lower())}\b",
            re.I,
        )
        if neg_pat.search(lower) and len(lower) < 200:
            flags.append("negation")
            return GateResult(False, "negated_target", tuple(flags))

    return GateResult(True, "ok", ())

