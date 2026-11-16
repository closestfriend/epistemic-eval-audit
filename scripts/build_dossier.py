#!/usr/bin/env python3
"""Build a self-contained, lightweight HTML dossier for reading benchmark tasks,
candidate outputs, deterministic gate decisions, and raw LLM judge reasoning.

Outputs: results/dossier.html (opens directly in any browser with no server needed).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.adaption_provider import _cache_key
from targets.darkbench_native import DARK_PATTERNS, OVERSEER_SYSTEM_MESSAGE, OVERSEER_TEMPLATE
from targets.mask_native import (
    BINARY_JUDGE_PROMPT_TEMPLATE,
    DISINFORMATION_RULES,
    parse_proposition,
)
from targets.syceval_native import SYCEVAL_SYSTEM_PROMPT, SYCEVAL_USER_TEMPLATE
from verifier.equivalence import check_degenerate_gates


def _load_module(name: str, rel_path: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_caches() -> dict[str, dict[str, dict[str, Any]]]:
    caches: dict[str, dict[str, dict[str, Any]]] = {}
    for jm in ("qwen3.5-397b", "glm-5.2", "gemma-4-31b-it"):
        idx: dict[str, dict[str, Any]] = {}
        p = ROOT / "results" / "cache" / f"{jm}.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("key"):
                        idx[rec["key"]] = rec
                except json.JSONDecodeError:
                    continue
        caches[jm] = idx
    return caches


def _native_prompt_and_sys(benchmark: str, task: dict[str, Any], out: str) -> tuple[str, str]:
    if benchmark == "mask":
        _, (p1, p2) = parse_proposition(task["proposition"])
        rules = DISINFORMATION_RULES if "disinformation" in task.get("split", "").lower() else ""
        prompt = BINARY_JUDGE_PROMPT_TEMPLATE.format(
            system_prompt=task.get("system_prompt", "You are a helpful assistant."),
            user_prompt=task.get("user_prompt") or task.get("prompt", ""),
            model_response=out,
            proposition_option_1=p1,
            proposition_option_2=p2,
            rules=rules,
        )
        return "You are a helpful assistant.", prompt
    if benchmark == "syceval":
        prompt = SYCEVAL_USER_TEMPLATE.format(
            question=task.get("prompt") or task.get("question", ""),
            correct_answer=task.get("target") or task.get("correct_answer", ""),
            ai_generated_answer=out,
        )
        return SYCEVAL_SYSTEM_PROMPT, prompt
    issue_key = (task.get("target") or "sycophancy").replace("_", "-")
    if issue_key not in DARK_PATTERNS:
        issue_key = "sycophancy"
    dname, desc = DARK_PATTERNS[issue_key]
    prompt = OVERSEER_TEMPLATE.format(
        issue_key=issue_key,
        issue_display_name=dname,
        issue_description=desc,
        prompt=task.get("input") or task.get("prompt", ""),
        response=out,
    )
    return OVERSEER_SYSTEM_MESSAGE, prompt


def _lookup_cached(
    idx: dict[str, dict[str, Any]],
    model: str,
    sys_msg: str,
    prompt: str,
) -> dict[str, Any] | None:
    for tok in (2048, 300, 128, 256, 1024):
        k = _cache_key(model, sys_msg, prompt, 0.0, tok)
        if k in idx:
            return idx[k]
    return None


def _task_context(benchmark: str, task: dict[str, Any]) -> dict[str, Any]:
    if benchmark == "mask":
        _, (p1, p2) = parse_proposition(task.get("proposition", ""))
        return {
            "split": task.get("split", ""),
            "system_prompt": task.get("system_prompt", ""),
            "user_prompt": task.get("user_prompt") or task.get("prompt", ""),
            "ground_truth": f"{task.get('formatted_ground_truth', '')}: {p1 if (task.get('formatted_ground_truth') or 'A').upper() == 'A' else p2}",
            "prop_a": p1,
            "prop_b": p2,
        }
    if benchmark == "syceval":
        return {
            "split": task.get("domain", ""),
            "system_prompt": "",
            "user_prompt": task.get("prompt") or task.get("question", ""),
            "ground_truth": str(task.get("target") or task.get("correct_answer", "")),
            "rebuttal": task.get("rebuttal", ""),
        }
    return {
        "split": task.get("target", ""),
        "system_prompt": "",
        "user_prompt": task.get("input") or task.get("prompt", ""),
        "ground_truth": f"No {task.get('target', 'dark pattern')}",
    }


def build_meta_cases(caches: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    m02 = _load_module("m02", "experiments/02_run_judge_meta.py")
    items = m02.build_meta_dataset(85, 42)

    base_csv: dict[str, dict[str, str]] = {}
    hard_csv: dict[str, dict[str, str]] = {}
    with open(ROOT / "results" / "judge_meta_baseline.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            base_csv[r["item_id"]] = r
    hard_path = ROOT / "results" / "judge_meta_hardened.csv"
    if hard_path.exists():
        with open(hard_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                hard_csv[r["item_id"]] = r

    # Track per-benchmark index to tag mode0/mode1/mode2
    counts: dict[str, int] = {"mask": 0, "syceval": 0, "darkbench": 0}
    mode_names = ["mode0_positive", "mode1_negative", "mode2_evasive"]
    cases: list[dict[str, Any]] = []

    for it in items:
        iid = it["item_id"]
        b = it["benchmark"]
        idx_in_b = counts[b]
        counts[b] += 1
        stratum = mode_names[idx_in_b % 3]
        task = it["task"]
        out = it["candidate_output"]
        sys_msg, prompt = _native_prompt_and_sys(b, task, out)
        gate = check_degenerate_gates(out, task)

        b_row = base_csv.get(iid, {})
        h_row = hard_csv.get(iid, {})

        judges = []
        csv_cols = [
            ("qwen3.5-397b", "judge"),
            ("glm-5.2", "judge2"),
            ("gemma-4-31b-it", "judge3"),
        ]
        for jm, col in csv_cols:
            rec = _lookup_cached(caches[jm], jm, sys_msg, prompt)
            judges.append({
                "model": jm,
                "native_pass": b_row.get(col, ""),
                "hardened_pass": h_row.get(col, ""),
                "native_reasoning": rec.get("text", "") if rec else "",
                "output_tokens": rec.get("output_tokens", 0) if rec else 0,
            })

        cases.append({
            "id": iid,
            "kind": "meta",
            "benchmark": b,
            "stratum": stratum,
            "spec_label": it["reference"],
            "context": _task_context(b, task),
            "candidate_output": out,
            "gate": {
                "passed": gate.passed_gate,
                "reason": gate.reason,
                "flags": list(gate.flags),
            },
            "judges": judges,
        })
    return cases


def build_redteam_cases(caches: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    m01 = _load_module("m01", "experiments/01_run_redteam.py")
    suites = [
        ("mask", m01.load_stratified_mask(35)),
        ("syceval", m01.load_stratified_syceval(35)),
        ("darkbench", m01.load_stratified_darkbench(30)),
    ]
    cases: list[dict[str, Any]] = []
    # Include 3 representative tasks per benchmark x all 16 attacks (144 rich examples)
    for b, tasks in suites:
        for t_idx, task in enumerate(tasks[:3]):
            for atk_name, atk_body in m01.build_task_attacks(task):
                sys_msg, prompt = _native_prompt_and_sys(b, task, atk_body)
                rec = _lookup_cached(caches["gemma-4-31b-it"], "gemma-4-31b-it", sys_msg, prompt)
                gate = check_degenerate_gates(atk_body, task)
                cases.append({
                    "id": f"rt_{b}_{t_idx:02d}_{atk_name}",
                    "kind": "redteam",
                    "benchmark": b,
                    "stratum": atk_name,
                    "spec_label": "0",
                    "context": _task_context(b, task),
                    "candidate_output": atk_body,
                    "gate": {
                        "passed": gate.passed_gate,
                        "reason": gate.reason,
                        "flags": list(gate.flags),
                    },
                    "judges": [
                        {
                            "model": "gemma-4-31b-it",
                            "native_pass": "",
                            "hardened_pass": "0" if not gate.passed_gate else "",
                            "native_reasoning": rec.get("text", "") if rec else "",
                            "output_tokens": rec.get("output_tokens", 0) if rec else 0,
                        }
                    ],
                })
    return cases


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>Epistemic Eval Audit — Output Dossier</title>
  <style>
    :root {
      --bg-light: #f7f7f5;
      --bg-dark: #111315;
      --surface-light: #ffffff;
      --surface-dark: #191c20;
      --surface-alt-light: #f0f0ec;
      --surface-alt-dark: #21252b;
      --border-light: #deded8;
      --border-dark: #2d323a;
      --text-light: #181a1c;
      --text-dark: #e8eaed;
      --muted-light: #5c636a;
      --muted-dark: #9aa0a6;
      --accent-light: #1d4ed8;
      --accent-dark: #7aa2f7;
      --pass-light: #15803d;
      --pass-dark: #4ade80;
      --pass-bg-light: #dcfce7;
      --pass-bg-dark: #143321;
      --fail-light: #b91c1c;
      --fail-dark: #f87171;
      --fail-bg-light: #fee2e2;
      --fail-bg-dark: #3b1818;
      --warn-light: #92400e;
      --warn-dark: #fbbf24;
      --warn-bg-light: #fef3c7;
      --warn-bg-dark: #362a11;

      --bg: var(--bg-light);
      --surface: var(--surface-light);
      --surface-alt: var(--surface-alt-light);
      --border: var(--border-light);
      --text: var(--text-light);
      --muted: var(--muted-light);
      --accent: var(--accent-light);
      --pass: var(--pass-light);
      --pass-bg: var(--pass-bg-light);
      --fail: var(--fail-light);
      --fail-bg: var(--fail-bg-light);
      --warn: var(--warn-light);
      --warn-bg: var(--warn-bg-light);

      @media (prefers-color-scheme: dark) {
        --bg: var(--bg-dark);
        --surface: var(--surface-dark);
        --surface-alt: var(--surface-alt-dark);
        --border: var(--border-dark);
        --text: var(--text-dark);
        --muted: var(--muted-dark);
        --accent: var(--accent-dark);
        --pass: var(--pass-dark);
        --pass-bg: var(--pass-bg-dark);
        --fail: var(--fail-dark);
        --fail-bg: var(--fail-bg-dark);
        --warn: var(--warn-dark);
        --warn-bg: var(--warn-bg-dark);
      }

      @supports (color: light-dark(white, black)) {
        --bg: light-dark(var(--bg-light), var(--bg-dark));
        --surface: light-dark(var(--surface-light), var(--surface-dark));
        --surface-alt: light-dark(var(--surface-alt-light), var(--surface-alt-dark));
        --border: light-dark(var(--border-light), var(--border-dark));
        --text: light-dark(var(--text-light), var(--text-dark));
        --muted: light-dark(var(--muted-light), var(--muted-dark));
        --accent: light-dark(var(--accent-light), var(--accent-dark));
        --pass: light-dark(var(--pass-light), var(--pass-dark));
        --pass-bg: light-dark(var(--pass-bg-light), var(--pass-bg-dark));
        --fail: light-dark(var(--fail-light), var(--fail-dark));
        --fail-bg: light-dark(var(--fail-bg-light), var(--fail-bg-dark));
        --warn: light-dark(var(--warn-light), var(--warn-dark));
        --warn-bg: light-dark(var(--warn-bg-light), var(--warn-bg-dark));
      }

      color-scheme: light dark;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    header {
      padding: 12px 20px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
    }

    .title-group h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 600;
      letter-spacing: -0.01em;
    }
    .title-group .subtitle {
      font-size: 12px;
      color: var(--muted);
    }

    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }

    select, input[type="search"] {
      background: var(--surface-alt);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 13px;
    }
    input[type="search"] { width: 220px; }

    .count-badge {
      font-size: 12px;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      padding: 0 6px;
    }

    main {
      display: grid;
      grid-template-columns: 340px 1fr;
      flex: 1;
      min-height: 0;
    }

    .list-pane {
      border-right: 1px solid var(--border);
      background: var(--surface);
      overflow-y: auto;
    }

    .case-item {
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      cursor: pointer;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .case-item:hover { background: var(--surface-alt); }
    .case-item.active {
      background: var(--surface-alt);
      border-left: 3px solid var(--accent);
      padding-left: 11px;
    }
    .case-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      font-weight: 600;
    }
    .case-preview {
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .reader-pane {
      overflow-y: auto;
      padding: 24px 32px 64px;
    }
    .reader-inner {
      max-width: 82ch;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px 18px;
    }
    .card-label {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    pre.prose {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
      line-height: 1.55;
    }

    .pill {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      font-variant-numeric: tabular-nums;
    }
    .pill.pass { background: var(--pass-bg); color: var(--pass); }
    .pill.fail { background: var(--fail-bg); color: var(--fail); }
    .pill.warn { background: var(--warn-bg); color: var(--warn); }
    .pill.neutral { background: var(--surface-alt); color: var(--muted); }

    .judge-grid {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .judge-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
    }
    .judge-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      gap: 8px;
      flex-wrap: wrap;
    }
    .judge-name { font-weight: 600; font-size: 13px; }
    .verdicts { display: flex; gap: 6px; align-items: center; }

    .kbd-hint {
      font-size: 11px;
      color: var(--muted);
    }

    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; grid-template-rows: 220px 1fr; }
      .list-pane { border-right: none; border-bottom: 1px solid var(--border); }
    }
  </style>
</head>
<body>
  <header>
    <div class="title-group">
      <h1>Epistemic Eval Audit — Output Dossier</h1>
      <div class="subtitle">Inspect raw prompts, candidate outputs, deterministic gates, and judge reasoning (<span class="kbd-hint">j / k to step</span>)</div>
    </div>
    <div class="controls">
      <select id="kindFilter" aria-label="Dataset">
        <option value="all">All Cases</option>
        <option value="meta" selected>Meta-Eval (255)</option>
        <option value="redteam">Red-Team Samples (144)</option>
      </select>
      <select id="benchFilter" aria-label="Benchmark">
        <option value="all">All Benchmarks</option>
        <option value="mask">MASK</option>
        <option value="syceval">SycEval</option>
        <option value="darkbench">DarkBench</option>
      </select>
      <select id="stratumFilter" aria-label="Stratum">
        <option value="all">All Strata</option>
        <option value="mode0_positive">mode0 (Positive / Clean)</option>
        <option value="mode1_negative">mode1 (Negative / Lie)</option>
        <option value="mode2_evasive">mode2 (Evasive / Refusal)</option>
        <option value="disagree">Judges Disagree</option>
        <option value="nat_vs_hard">Native != Hardened</option>
      </select>
      <input id="searchBox" type="search" placeholder="Filter text, ID, attack...">
      <span id="countBadge" class="count-badge">0 items</span>
    </div>
  </header>

  <main>
    <aside id="listPane" class="list-pane" role="listbox" aria-label="Cases"></aside>
    <section id="readerPane" class="reader-pane" aria-live="polite"></section>
  </main>

  <script id="dossier-data" type="application/json">__DOSSIER_JSON__</script>
  <script>
    const ALL_CASES = JSON.parse(document.getElementById("dossier-data").textContent);
    const kindFilter = document.getElementById("kindFilter");
    const benchFilter = document.getElementById("benchFilter");
    const stratumFilter = document.getElementById("stratumFilter");
    const searchBox = document.getElementById("searchBox");
    const listPane = document.getElementById("listPane");
    const readerPane = document.getElementById("readerPane");
    const countBadge = document.getElementById("countBadge");

    let filtered = [];
    let selectedIdx = 0;

    function esc(str) {
      return String(str ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function pill(label, cls) {
      return `<span class="pill ${cls}">${esc(label)}</span>`;
    }

    function passPill(prefix, val) {
      if (val === "1") return pill(`${prefix}: PASS`, "pass");
      if (val === "0") return pill(`${prefix}: FAIL`, "fail");
      return pill(`${prefix}: —`, "neutral");
    }

    function judgesDisagree(c) {
      const vals = c.judges.map(j => j.native_pass).filter(v => v !== "");
      return new Set(vals).size > 1;
    }

    function natVsHardDiff(c) {
      return c.judges.some(j => j.native_pass !== "" && j.hardened_pass !== "" && j.native_pass !== j.hardened_pass);
    }

    function applyFilters() {
      const k = kindFilter.value;
      const b = benchFilter.value;
      const s = stratumFilter.value;
      const q = searchBox.value.trim().toLowerCase();

      filtered = ALL_CASES.filter(c => {
        if (k !== "all" && c.kind !== k) return false;
        if (b !== "all" && c.benchmark !== b) return false;
        if (s === "disagree" && !judgesDisagree(c)) return false;
        else if (s === "nat_vs_hard" && !natVsHardDiff(c)) return false;
        else if (s !== "all" && s !== "disagree" && s !== "nat_vs_hard" && c.stratum !== s) return false;
        if (q) {
          const hay = (
            c.id + " " + c.stratum + " " +
            (c.context.user_prompt || "") + " " +
            (c.context.ground_truth || "") + " " +
            c.candidate_output + " " +
            c.judges.map(j => j.native_reasoning).join(" ")
          ).toLowerCase();
          if (!hay.includes(q)) return false;
        }
        return true;
      });

      countBadge.textContent = `${filtered.length} items`;
      selectedIdx = 0;
      renderList();
      renderReader();
    }

    function renderList() {
      listPane.innerHTML = filtered.map((c, idx) => {
        const j0 = c.judges[0] || {};
        const activeCls = idx === selectedIdx ? "active" : "";
        const natSum = c.judges.map(j => j.native_pass || "?").join("/");
        const hardSum = c.judges.map(j => j.hardened_pass || "?").join("/");
        return `
          <div class="case-item ${activeCls}" data-idx="${idx}" role="option" aria-selected="${idx === selectedIdx}">
            <div class="case-top">
              <span>${esc(c.id)} <span class="pill neutral">${esc(c.stratum)}</span></span>
              <span>N:${esc(natSum)} H:${esc(hardSum)}</span>
            </div>
            <div class="case-preview">${esc(c.context.user_prompt || c.candidate_output)}</div>
          </div>
        `;
      }).join("");

      listPane.querySelectorAll(".case-item").forEach(el => {
        el.addEventListener("click", () => {
          selectedIdx = Number(el.dataset.idx);
          renderList();
          renderReader();
        });
      });
    }

    function renderReader() {
      if (!filtered.length) {
        readerPane.innerHTML = `<div class="reader-inner"><div class="card">No matching items.</div></div>`;
        return;
      }
      const c = filtered[selectedIdx];
      const ctx = c.context || {};
      const gatePill = c.gate.passed
        ? pill("Gate: PASSED", "pass")
        : pill(`Gate: REJECTED (${c.gate.reason})`, "warn");

      const sysBlock = ctx.system_prompt
        ? `<div class="card"><div class="card-label">System Prompt</div><pre class="prose">${esc(ctx.system_prompt)}</pre></div>`
        : "";

      const extraTruth = ctx.prop_a
        ? `\nA: ${ctx.prop_a}\nB: ${ctx.prop_b}`
        : (ctx.rebuttal ? `\nRebuttal: ${ctx.rebuttal}` : "");

      const judgesHtml = c.judges.map(j => `
        <div class="judge-card">
          <div class="judge-header">
            <span class="judge-name">${esc(j.model)}</span>
            <div class="verdicts">
              ${passPill("Native", j.native_pass)}
              ${passPill("Hardened", j.hardened_pass)}
              ${j.output_tokens ? pill(`${j.output_tokens} tok`, "neutral") : ""}
            </div>
          </div>
          <pre class="prose">${esc(j.native_reasoning || "(No cached native reasoning text for this entry)")}</pre>
        </div>
      `).join("");

      readerPane.innerHTML = `
        <div class="reader-inner">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
            <div>
              <strong style="font-size:16px;">${esc(c.id)}</strong>
              ${pill(c.benchmark.toUpperCase(), "neutral")}
              ${pill(c.stratum, "neutral")}
              ${pill("spec_label=" + c.spec_label, c.spec_label === "1" ? "pass" : "fail")}
            </div>
            <div>${gatePill}</div>
          </div>

          ${sysBlock}

          <div class="card">
            <div class="card-label"><span>User Prompt</span><span>Split: ${esc(ctx.split || "—")}</span></div>
            <pre class="prose">${esc(ctx.user_prompt)}</pre>
          </div>

          <div class="card">
            <div class="card-label"><span>Ground Truth / Target</span></div>
            <pre class="prose">${esc((ctx.ground_truth || "") + extraTruth)}</pre>
          </div>

          <div class="card">
            <div class="card-label"><span>Candidate Output</span><span>${c.candidate_output.length} chars</span></div>
            <pre class="prose">${esc(c.candidate_output || "(empty string)")}</pre>
          </div>

          <div>
            <div class="card-label" style="margin-bottom:8px;"><span>LLM Judge Verdicts & Raw Reasoning</span></div>
            <div class="judge-grid">${judgesHtml}</div>
          </div>
        </div>
      `;
    }

    [kindFilter, benchFilter, stratumFilter].forEach(el => el.addEventListener("change", applyFilters));
    searchBox.addEventListener("input", applyFilters);

    document.addEventListener("keydown", e => {
      if (document.activeElement === searchBox) return;
      if (e.key === "j" || e.key === "ArrowDown") {
        if (selectedIdx < filtered.length - 1) {
          selectedIdx++;
          renderList();
          renderReader();
          listPane.querySelector(".case-item.active")?.scrollIntoView({ block: "nearest" });
        }
      } else if (e.key === "k" || e.key === "ArrowUp") {
        if (selectedIdx > 0) {
          selectedIdx--;
          renderList();
          renderReader();
          listPane.querySelector(".case-item.active")?.scrollIntoView({ block: "nearest" });
        }
      }
    });

    applyFilters();
  </script>
</body>
</html>
"""


def main() -> None:
    caches = _load_caches()
    cases = build_meta_cases(caches) + build_redteam_cases(caches)
    payload = json.dumps(cases, ensure_ascii=False).replace("</script>", "<\\/script>")
    out_path = ROOT / "results" / "dossier.html"
    out_path.write_text(HTML_TEMPLATE.replace("__DOSSIER_JSON__", payload), encoding="utf-8")
    print(f"Wrote {out_path} ({len(cases)} cases, {out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

