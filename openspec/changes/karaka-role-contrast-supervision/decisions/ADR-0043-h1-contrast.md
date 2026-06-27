# ADR-0043 — Pre-register H1_CONTRAST (gold kāraka role-contrast supervision)

> **Canonical home:** `docs/decisions/ADR-0043-h1-contrast.md`. It lives inside
> this change while `docs/` is gitignored on the code-only branch; copy it to
> `docs/decisions/` on the docs-tracked branch and confirm the number against the
> ADR registry (0042 = arch bake-off was the last known).

- **Status:** Proposed — pre-registration, pending human sign-off (closure layer 6)
- **Date:** 2026-06-27
- **Change:** `openspec/changes/karaka-role-contrast-supervision/`
- **Relates to:** ADR-0017 (H1_COGS closed null), ADR-0038/0039 (H1_MECHANISM),
  ADR-0033 (Vidyut realizer)

## Context

Every Pāṇinian role mechanism tried so far has returned null, and the nulls are
**confounded**, which is why they are uninterpretable rather than informative:

- **F2 (masking):** kāraka-stratified masking causally inert at matched budget
  (Δ = +0.10, ns).
- **F3 (auxiliary):** role head gives +0.76 targeted points / 5 seeds (ns),
  attenuating with each seed.
- **F8 (v0.2 masking):** the deprel/MI arms ran *byte-identical to control* —
  dead wiring, never a real test.

Two confounds run through all of them: (1) **label noise** — English roles are
guessed by spaCy/heuristics, the paper's own first-named limitation; and (2) a
**broken injection point** — the per-vocab masking table cannot carry per-instance
roles. We cannot tell whether the prior failed or the plumbing did.

`H1_CONTRAST` removes both confounds. Gold kāraka labels are minted on the
Sanskrit side (Vidyut, by rule — verified, see gate 1.1) and the **same kāraka
frame is realized into both Sanskrit and English** (Route A; gate 1.3 found no
attested parallel data in-repo), so English roles are **gold by construction, with
no translation and no alignment**. The signal is injected through the supervised
auxiliary role-prediction head (the F3 path), not mask biasing. Supervision is
delivered as **role-contrast sets**: one event, one role held invariant, surface
varied across constructions (active ↔ passive ↔ case-frames), teaching
role-invariance directly.

This is a distinct hypothesis from H1_MECHANISM (masking; frozen arms) and
H1_COGS (synthetic dose; closed null). It does not reopen either.

## Decision

We pre-register `H1_CONTRAST` with the following protocol, fixed before any run.

**Hypothesis.** Gold kāraka role-contrast supervision (Route A bilingual
realization), injected via the supervised auxiliary role-prediction head,
improves English argument-role generalization over a matched baseline.

**Independent variable.** Presence of gold role-contrast supervision. Everything
else held equal.

**Matched baseline + specificity control.** The contrast arm is compared to:
1. a **matched baseline** — identical budget, identical aux-head capacity, no
   role-contrast signal; and
2. a **shuffled-role control** — the same aux head fed labels that preserve the
   role *distribution* but destroy token alignment. The effect counts only if
   `real − shuffled > 0` and significant (mirrors F3's specificity control;
   rules out generic multi-task regularization).

**Primary metric + threshold (locked at sign-off).**
- **Primary surface:** the **COGS argument-role subset** — `active_to_passive`,
  `passive_to_active`, `do_dative_to_pp_dative`, `pp_dative_to_do_dative` — which
  `cogs.py` itself identifies as "where the mechanism predicts" a benefit and
  which matches the contrast sets directly.
- **Threshold:** ≥ **+3.0** points exact-match generalization accuracy on the
  argument-role subset vs. the matched baseline, AND `real − shuffled` positive
  and significant.
- **Corroborating surface:** BLiMP targeted subset (agreement +
  argument-structure + passive/dative), pre-registered secondary threshold
  ≥ **+1.0** point (mirrors the short-paper pre-registration).

**Statistics.** ≥3 seeds (≥5 if the primary lands marginal), mean ± 95% CI,
paired bootstrap over per-category subsets, Holm–Bonferroni across the metric
family. A single-seed result is preliminary per the closure rule.

**Closure / honesty rules (binding).** No "failed" on attempt 1; a NULL requires
≥2 documented interventions (e.g. alignment-free gloss expansion, construction
mix, aux weight λ) logged in the ledger with attempt #, change, result,
interpretation. Coverage (frames/constructions attempted, retained, dropped) is
logged — no silent truncation.

**Scope.** This ADR pre-registers **Route A only** (synthetic bilingual
realization → *internal* validity: does gold role-contrast move the metric at all,
free of label-noise and alignment confounds?). **Route B** (external natural
parallel corpus → *external* validity) is a separate future pre-registration with
its own threshold; its alignment-projected labels reintroduce noise this ADR
deliberately excludes.

## Alternatives considered

- **Mask-probability biasing** — rejected: the per-vocab socket is dead (F8) and
  cannot carry per-instance roles. The aux head is the working injection point.
- **spaCy-guessed English roles** — rejected: this *is* the label noise we are
  escaping; using it would re-confound the test.
- **pañcāvayava-generated data** — rejected: argument-level (H2), requires a
  vyāpti knowledge base (H3), and produces templated synthetic text. Out of scope.
- **Route B first (natural parallel + alignment)** — rejected as the *initial*
  test: alignment noise would confound the clean internal-validity question.
  Sequenced as the scale-up once Route A shows signal.

## Consequences

**Positive.**
- The first **interpretable** test of the Pāṇinian role prior: gold labels on both
  sides + correct injection point remove both confounds, so a null is strong
  ("even perfect gold contrast, zero alignment noise, didn't move it") and a win
  is real. This is the ADR's main contribution regardless of outcome.
- Makes Vidyut load-bearing in the English path for the first time.
- Reuses the F3 aux-head + `RoleStreamPacker` machinery; small build surface.

**Negative / risks.**
- Route A English is synthetic and lexically narrow (≈38 stems / 8 verbs), so a
  *win* may reflect template-learning, not transferable structure; external
  validity is unproven until Route B. Mitigated by keeping it auxiliary
  supervision (not a corpus) and by the planned Route-B follow-on.
- Small gold lexicon caps construction/vocabulary diversity; expandable via DCS-
  grounded frames + gloss authoring, not free.

**Neutral.**
- Reusing the COGS *surface* does not reopen H1_COGS (a different mechanism — the
  synthetic dose — measured on the same benchmark).

## Sign-off

Pending human sign-off on the hypothesis framing and the locked threshold values
(closure layer 6) before any code or run. Thresholds above are the proposed
pre-registration; they become binding at sign-off.
