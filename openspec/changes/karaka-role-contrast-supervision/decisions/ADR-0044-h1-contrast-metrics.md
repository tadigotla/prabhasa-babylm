# ADR-0044 — Re-designate H1_CONTRAST metrics (amends ADR-0043)

> **Canonical home:** `docs/decisions/ADR-0044-h1-contrast-metrics.md`. Lives inside
> this change while `docs/` is gitignored on the code-only branch; copy on the
> docs-tracked branch and confirm the number against the registry.

- **Status:** Proposed — pre-registration, **must be signed off BEFORE the
  full-corpus run** (else the metric choice is post-hoc). Closure layer 6.
- **Date:** 2026-06-27
- **Amends:** ADR-0043 (metric clause only; all else stands)
- **Evidence:** `results/task6-proxy-finding.md`, `results/task6-step1-probe-finding.md`

## Context

ADR-0043 locked the H1_CONTRAST primary metric as **COGS argument-role +3.0**, with
**BLiMP targeted +1.0** as corroborating. Two things since invalidate that framing
as written:

1. **COGS was not computable as locked.** The repo's COGS discrimination eval trains
   its *own* causal models; it is not a scorer for an MLM checkpoint. The proxy run
   (task 6) substituted BLiMP and logged the gap — an *undocumented metric
   substitution* that this ADR now regularizes.
2. **BLiMP is the wrong instrument for this hypothesis, and the proxy showed it.**
   BLiMP pseudo-log-likelihood tests *surface grammar*; the mechanism injects
   *semantic argument roles*. On the proxy, BLiMP read null on the targeted subset
   (gold−baseline −0.42), while a **linear argument-role probe (n=9)** found a
   significant, alignment-specific effect (gold−baseline +6.8, t=5.6, 9/9 seeds;
   gold−shuffled +3.8, t=3.0). The probe — currently *nobody's pre-registered
   metric* — is the only instrument that has detected the effect.

Design-review threads that drive this amendment: (a) BLiMP is surface-blind to roles
and underpowered for the specificity contrast (the probe needed n=9 to clear zero —
n=3 BLiMP cannot); (b) the BLiMP "targeted subset" included **subject-verb
agreement**, which is morphosyntax — role-irrelevant — and dragged the targeted
number down; (c) silently promoting BLiMP to primary is itself a pre-registered-
metric change requiring an ADR.

## Decision

Re-designate the H1_CONTRAST metric framework as follows (pre-registered).

**Primary (task) — COGS argument-role discrimination.** Honors the original locked
primary. Computed via a **to-be-built COGS-disc-for-MLM scorer** (a structured
readout over the frozen encoder for COGS argument-role generalization; reuses
`cogs.load_cogs`, `domain.eval.discrimination.CORRUPTIONS`, and the checkpoint /
`hidden_mlm` extraction in `probe_roles.py`). Threshold: **gold − baseline ≥ +3.0
points** on the argument-role gen subset, paired across seeds.

**Confirmatory + specificity — linear argument-role probe, n ≥ 9.** The powered,
low-variance instrument. Carries the **specificity gate**: **gold − shuffled > 0,
significant at n ≥ 9** (proxy: +3.8, t=3.0 — must hold/grow off the floor). Also
reports gold − baseline (proxy: +6.8). The probe is now an *explicit* pre-registered
metric.

**Corroborating (coarse) — BLiMP role-sensitive subset.** Paradigms where argument
*roles* are the variable: argument-structure, passive, causative, argument-drop,
inchoative. **Subject-verb agreement is EXCLUDED** (morphosyntax; role-irrelevant
confound). BLiMP answers only the coarse "did it lift off the chance floor and move
in the right direction"; it is **not** a pass/fail gate and **not** the specificity
test.

**Crux pre-check (gate to the full run) — construction-invariance probe.** Train the
role probe on active sentences, test on passive. gold's active→passive transfer
significantly above baseline is a *necessary condition* (the role subspace is
invariant to a surface change) before committing GPU-hours.

**Batch.** The full run uses **grad-accumulation to effective batch 256** (matched
across arms) — not the proxy's batch 32 — so metrics lift off the floor.

## Honest caveat (binding on interpretation)

For a method evaluated through an **MLM encoder**, "downstream task accuracy" is
inherently **probe-flavored**: linear probe → structured head → fine-tune is a
capacity continuum, and the MLM does not natively *do* a task. The COGS-disc-for-MLM
scorer is therefore a *structured* probe — chosen because it is the most
**task-shaped** (roles are needed to build the logical form) and **non-circular**
(roles *help* the task but are not the label) pre-registered option. A fine-tune
that predicts roles would be a higher-capacity probe and is circular. This caveat
must appear in the finding.

## Consequences

- The hypothesis is finally judged on instruments that can see it: a role-dependent
  task (COGS) + the powered probe, with BLiMP demoted to honest corroboration.
- Resolves the silent metric substitution (governance) and the surface-grammar /
  agreement confounds.
- Cost: building the COGS-disc-for-MLM scorer is dev work (~1–2 h), gating the run.

## Alternatives rejected

- **Keep BLiMP primary** — surface-blind to roles, underpowered for specificity,
  agreement-confounded. The proxy already demonstrated the failure.
- **Probe as the *primary*** — the probe almost cannot fail for gold (the aux trains
  roles to be decodable); it is confirmatory, not a "does it become a task" test.
- **Fine-tune a role head as the task** — circular (predicting the label); it is a
  higher-capacity probe, not a distinct task.

## Sign-off

Pending human sign-off on this re-designation and the +3.0 / specificity thresholds
**before any full-corpus result is observed.**
