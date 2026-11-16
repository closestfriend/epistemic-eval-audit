# Epistemic Metadata Framework v2.0 (2026-11-14)

**Provenance:** Supersedes v1 (`MY FRAMEWORK.txt`, file-dated 2025-04-14; author states composition predates the file; Zenodo DOI 10.5281/zenodo.24273844, reserved, resolves on publication). v1 is immutable — this document cites it, never rewrites it.
**What changed since v1 and why:** every change below is driven by evidence gathered 2025–2026, tagged [GRD] (grounded in repo data or literature). The v1 design space was correct; its *semantics* were not — this document fixes the semantics.

---

## 1. Design principles (evidence-driven)

1. **Confidence is a measurement, not an adjective.** v1's `[Medium-High Confidence]` is verbalized confidence, which is systematically miscalibrated (Xiong et al. 2024; Zhou et al. 2024) [GRD]. Fields carry calibrated scores and *how they were measured*.
2. **Epistemic state has three dissociated channels** — internal (probes/semantic entropy), verbalized self-report (P(True)), and attached metadata (v1's approach). They disagree, and the disagreement is informative (judge-meta data: inter-judge κ 0.81–0.95 while judge-vs-template-intent κ 0.31–0.37 — consistency is not accuracy). [GRD]
3. **Epistemic markers are spoofable.** Strategic abstention and false-balance hedging pass native MASK scoring at 100% (35/35 each, repo redteam) — the judge returns C and the not-lying metric credits it; judges penalize honest hedging up to −47% (EMBER, NAACL 2025). A marker is a *signal emitted into an adversarial environment*; its generator must carry its own attack-resistance profile. [GRD]
4. **A claim without a refutation condition is unfalsifiable decoration.** The corrigibility field (new; absent from all 2026 documentation standards surveyed) makes every claim falsifiable per-claim. [GRD]
5. **Dual-layer by construction.** Human-glance layer and machine-actionable layer emitted from one source: the glance layer uses short bracketed tags (**[GRD]**, **[INF]**, **[TST]** …) — typographically ordinary, instantly recognizable, and free of exotic-symbol baggage — while the machine layer carries the same codes as enum values (v1 had only the glance layer; model cards' lack of formal semantics is the field's known failure — CEUR survey). [GRD]

## 2. Claim taxonomy (v1 classes + 2026 extensions)

| Tag | Class | Definition | Grounding |
|---|---|---|---|
| **[GRD]** | Grounded | Directly verifiable; resolvable citation present | chain-of-custody standard (repo "17% Gap" concern) |
| **[CON]** | Consensus report | Mainstream view of a *named* community, with contestation noted | epistemic-diversity lit (arXiv:2510.04226) |
| **[PRJ]** | Projection | Forward-looking; stated basis; decay-prone | v1 PROJECTION, retained |
| **[INF]** | Inference | Model-derived; reasoning chain available for audit | v1 PERSONAL INFERENCE, generalized |
| **[SPC]** | Speculative | No evidential support claimed | v1 SPECULATIVE, retained |
| **[TST]** | Testimony-only | Reported from a source the model cannot verify | new — the key hallucination-control class |

## 3. Field schema (per claim)

```yaml
claim: "..."
class: GRD | CON | PRJ | INF | SPC | TST   # (rendered in prose as: grounded / consensus / projection / inference / speculative / testimony-only)
confidence: 0.82                  # calibrated score, NOT an adjective
confidence_method:                # which channel produced it, and how audited
  channel: semantic_entropy | probe | verbalized_trained | human_set
  attack_resistance:              # NEW: audit of the marker generator itself
    tested_against: [strategic_abstention, false_balance_hedging, sycophantic_meta_ack, judge_injection]
    false_pass_rate: 0.00         # on those attacks, with n
    variance_note: "nonce-wrapped judge calls are non-deterministic across runs; see LAB_NOTES F6.4"
as_of: 2026-11-14                 # temporal validity of THIS claim
decay: "re-check on next benchmark release"   # v1's temporal field, made operational
consensus:
  community: "ML eval research"   # v1 axis, now names whose consensus
  strength: 0.9
  contestation: "calibration-vs-accuracy tradeoff disputed"
provenance:                       # PROV-O aligned (Croissant 1.1 / W3C)
  wasDerivedFrom: [doi:10.xxxx, arxiv:2510.04226]
  wasAttributedTo: [model_id_or_human_id]
  verification: {resolved: true, method: crossref_lookup}   # citation rot check
corrigibility: "falsified if: <explicit refutation condition>"  # NEW, required
```

Machine layer: the same object emits as JSON-LD (`@context` schema.org + PROV-O) alongside the inline human layer. One source, two renderings.

## 4. Self-demonstrating worked examples (real, from this session)

```yaml
claim: "SycEval's native judge is keyed to correctness and passes degenerate outputs at 0% while passing clean answers at 100%."
class: GRD
confidence: 0.99
confidence_method: {channel: human_set, attack_resistance: {tested_against: [empty, refusal], false_pass_rate: 0.0, n: 1120}}
as_of: 2026-11-14
provenance: {wasDerivedFrom: "results/anchor_run.log + anchor_run2.log", wasAttributedTo: "03_run_anchors.py @ commit-of-record"}
corrigibility: "falsified if: any degenerate anchor passes native SycEval in a rerun with n≥40"
```

```yaml
claim: "The wiki overnight run's 274 clustered timeouts were an API-side slowdown."
class: INF   # NOT GRD — see correction commit 58e09f6
as_of: 2026-11-06
corrigibility: "was revised 2026-11-13: client network later found degraded; both causes produce the same signature"
```

The second example is the framework's whole thesis in one record: **the original doc put an inference [INF] in grounded [GRD] position; the corrigibility field is what made the revision auditable rather than silent.**

## 5. What v1 got right that v2 preserves

Glance-level class tags for cognitive sorting (v2 uses bracketed letter codes rather than decorative glyphs — same function, interoperable form); separation of confidence from consensus from temporal context; the meta-note (self-reference) — now upgraded from a footnote to the audit requirement (§3 `attack_resistance`).

## 6. Open questions ([SPC], flagged)

- Should `confidence` aggregate channels when they disagree, or expose all three? (Lean: expose all three; disagreement is signal.)
- Human-set vs model-set marker reliability: predicted [GRD]-label calibration advantage for human annotators (testable in the evasion-prompt arm).
- Does per-claim marking change *user* trust calibration? Untested — the 2025 "Benefits" section remains an empirical question, not a feature list.

## Change log

- v2.0 (2026-11-14): calibrated scores replace adjectives; three-channel model; attack-resistance audit fields; corrigibility required; consensus community-named; PROV-O provenance with verification; testimony-only class added; dual-layer encoding spec. Driven by: repo redteam data, judge-meta data, anchor runs, EMBER, UQ surveys, CEUR metadata survey.
