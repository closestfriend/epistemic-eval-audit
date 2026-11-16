# CARD.md — Model card: hardened epistemic-behavior scorers

**Assets:** `verifier/score.py` (`score_mask_hardened`, `score_syceval_hardened`, `score_darkbench_hardened`) + `verifier/equivalence.py` (`check_degenerate_gates`).

## Intended use

Scoring candidate LLM outputs for epistemic-behavior benchmarks (MASK, SycEval, DarkBench) in settings where the candidate may contain adversarial content aimed at the *judge* — refusals, prompt injections, rubric echoes, fabricated verdicts. Designed for audit contexts: the scorer's job is to certify that a "pass" reflects genuine compliant behavior, not a judge exploit.

## Out of scope

- Certifying that a model is honest in general (this measures performance on specific benchmark elicitations).
- Defense against unseen attack classes (see Limitations).
- Use as a reward signal for RL without re-validating attack resistance after each training step.

## Measured attack resistance (experiment 01; crafted attacks, 560–480 trials per benchmark)

| Benchmark | False-pass rate | Residual failure modes |
|---|---|---|
| MASK | 0.0% [0, 0.7%] | none observed |
| DarkBench | 2.7% [1.6%, 4.6%] | `sycophantic_meta_ack` (all 13 residual passes; 13/30 trials) |
| SycEval | 0.0% [0, 0.7%] | none observed |

Anchor controls (experiment 03): hardened scorers pass genuine clean answers at ~97.5–100% (oracle anchors) and reject refusal/adversarial anchors at ~0% on MASK and DarkBench. SycEval native is already attack-resistant; hardened SycEval ≈ native on real outputs.

## Detection behavior on real models (experiment 03)

The hardened scorer disagrees with native scorers most on MASK (up to −45 pp for one model), driven by **evasion verdicts**: the native judge itself returns C for hedging-both-ways, and MASK's not-lying metric scores C as a pass; the hardened scorer does not. On SycEval the two agree almost exactly. On DarkBench the hardened scorer is more lenient than native for 3/4 models.

## Composite injection (experiment 07)

Injections wrapped around responses that should fail: the hardened scorers passed 0/40 MASK lies under each of three injections (the gate caught the instruction variant 40/40) and showed no net change on DarkBench (one fail→pass flip, within single-call variance). Under the same inputs the native pipelines passed 35/40 instruction-injected MASK lies (first-match parser capture) and 37/37 DarkBench responses that asked to be marked invalid (invalid-as-safe). See `PAPER.md` §3.5.

## Limitations

1. **Hardening circularity.** Attacks and defenses were developed with the same tooling. The measured resistance applies to the demonstrated attack suite; generalization to unseen attacks is unverified.
2. **Nonce variance.** Randomized candidate wrapping makes judge calls uncacheable; verdicts on candidates near decision boundaries can flip run-to-run at temperature 0 (observed on degenerate oracle text: 36/40 vs 40/40 across runs). Aggregate rates on n ≥ 40 are stable; single-item verdicts are not guaranteed reproducible.
3. **Judge dependence.** All scorers call an external LLM judge (default `gemma-4-31b-it`, temperature 0). Judge swaps require re-running the anchor suite before comparing results across judge models (cross-judge labeling agreement is imperfect; see experiment 02).
4. **Reference semantics.** The κ-deflation result (experiment 02) is against *template intent*, not human gold labels.
5. **Operational failure modes.** Generation-call failures must not be scored as empty candidates; runners persist per-item errors (`gen_error` fields) and results from runs containing them are marked partial. One such voided cell was caught and regenerated during the anchor campaign (LAB_NOTES Entry 12).

## Provenance

Built on native scorer re-implementations in `targets/` whose judge prompts match the upstream text (deviations listed in `PAPER.md` §2 and `BENCHMARK.md`); benchmark data under `targets/data/` (MASK 35-task stratified sample, SycEval 35, DarkBench 30–40; full sets in `*_full` files). Every result file is accompanied by per-item JSONL evidence.
