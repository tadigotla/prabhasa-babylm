# H1_CONTRAST — Step 0: construction-invariance probe (the crux gate)

**Date:** 2026-06-27 · **Hardware:** NVIDIA L4 (resume AMI, 27 checkpoints) · **Status:**
CRUX GATE **DOES NOT PASS** — strong construction-invariance fails; the n=9 effect is
**mostly surface/position-tied**, with a small (+1.6) significant *invariant residual*.

## Why this exists

The n=9 role probe (`task6-step1-probe-finding.md`) found gold makes argument roles
**+6.8 points** more linearly decodable from the frozen encoder. But that probe trained
and tested on the *same surface* (COGS train → COGS gen, both predominantly active). It
could not tell a genuine **role** axis from a **position** axis: in active English the
subject *is* the agent, so "first argument = agent" scores high without encoding any role
abstraction. Step 0 is the pre-registered crux pre-check (ADR-0044): **is gold's role
subspace invariant to a surface voice change (active ↔ passive)?** That invariance is a
*necessary condition* for the downstream-task and cross-lingual claims.

## Method

Data is **COGS only** — held out from every arm (no arm ever trained on COGS), so the
test is fair. The gold/shuffled arms *did* see the contrast active/passive realizations
via the aux head, so testing invariance on those would be circular; COGS avoids that.

Roles are labeled from the COGS logical form (agent/theme/recipient); extraction
(`sent_examples` / `collect` / linear probe / 400 steps / probe seed 0) is byte-identical
to `probe_roles.py`, so numbers are comparable. Primary metric: **agent-vs-theme
accuracy** (the kartā/karma distinction; balanced → **50% chance floor**). Two probes,
9 seeds × 3 arms = 27 checkpoints each.

- **Probe 1 — strong crux (`scripts/probe_invariance.py`).** Train on simple ACTIVES
  ("NP VERBed NP"), test on simple long PASSIVES ("NP was VERBed by NP", agent in the
  by-phrase). Same roles, surface positions flipped. A position-tied probe inverts on
  passives (calls the theme-subject "agent") and scores *below* chance.
- **Probe 2 — fair test (`scripts/probe_invariance_mixed.py`).** Train on a **balanced**
  mix (520 actives + 520 passives), so position no longer predicts role; test on held-out
  passives. This isolates whether a *position-robust* role axis exists at all, and whether
  gold encodes it better.

## Results

**Probe 1 — train ACTIVE → test PASSIVE (agent-vs-theme, chance 50):**

```
                         baseline   gold   shuffled
active → PASSIVE (crux)    12.09    8.95    13.72     ← ALL far below chance (50)
active → active (ceiling)  93.03   96.16    93.90
  crux  gold−baseline  −3.14  CI[−8.2,+2.0]  t=−1.42  3/9 +   (n.s., wrong direction)
  crux  gold−shuffled  −4.77  CI[−10.0,+0.4] t=−2.11  2/9 +
  ceil  gold−baseline  +3.13  CI[+0.6,+5.7]  t=2.82   7/9 +   (reproduces n=9 within-surface)
```

Every arm collapses to ~6–22% on passives — **systematic positional inversion**, not
noise (noise would sit at 50%). Gold transfers no better than baseline (slightly worse);
gold is the *most* positional (highest active→active ceiling, lowest active→passive).

**Probe 2 — balanced mixed-voice training → held-out PASSIVE (agent-vs-theme):**

```
                         baseline   gold   shuffled
held-out PASSIVE           95.27   96.83    95.61     ← ALL ≫ chance once both voices seen
held-out ACTIVE            81.52   87.05    82.36
  passive gold−baseline  +1.56  CI[+0.3,+2.8]  t=2.79  7/9 +   (significant, but small)
  passive gold−shuffled  +1.22  CI[+0.4,+2.1]  t=3.36  8/9 +   (alignment-specific)
  active  gold−baseline  +5.53  CI[+1.2,+9.8]  t=2.96  8/9 +
```

A **position-robust** agent/theme axis exists in *every* arm (held-out passives ≈ 95%+).
Gold encodes it **+1.56** better than baseline (significant, alignment-specific). But the
margin is small and near a 95% ceiling.

## Finding: representational, **mostly surface-tied**; the strong crux fails

1. **The strong construction-invariance crux does not pass.** An active-trained linear
   role axis collapses below chance on passives for *all* arms, gold included. The
   decodability the n=9 probe measured is **dominated by surface position**, not an
   abstract role code that reads off across a voice change. Per the pre-registered Step-0
   rule ("if it collapses to baseline, the +6.8 was surface-tied — reconsider before
   spending GPU-hours"), this is a **STOP-and-report**.

2. **A small invariant residual is real.** Once position is decorrelated (mixed training),
   gold's advantage survives at **+1.6 points** on held-out passives — significant and
   alignment-specific (+1.2 vs shuffled). So gold *does* sharpen a genuinely
   construction-invariant role axis, just much less than the headline n=6.8 implied.

3. **Re-reading the n=9 result.** Gold's role-decodability advantage decomposes into a
   large **position-conditioned** part (it reads roles better *given the active surface*)
   and a small **position-invariant** part. Going from same-surface to
   position-decorrelated shrinks the gold−baseline gap from **+6.8 → +1.6**. The mechanism
   helps roles be read off in context far more than it builds a transferable role
   abstraction.

## Tarka memo (strongest objection to my own STOP, and its resolution)

**Objection.** "The active-only → passive test is an unfairly hard bar that *no* linear
probe can pass: in active-only English, position perfectly predicts role, so a linear
probe never needs an invariant axis even if one exists. 'Gold fails the strong crux' is
therefore not evidence against gold — it is evidence the bar is unpassable. The *fair*
test is the mixed-voice one, and there gold **wins** significantly (+1.6, alignment-
specific). The lead is alive; STOP is too pessimistic."

**Resolution.** The objection is partly right — which is exactly why Probe 2 was run
rather than declaring failure on Probe 1 alone. The honest synthesis: gold's invariant
advantage is **real but small** (+1.6 at a 95% ceiling). The original green-light
rationale was that roles are encoded abstractly *enough* to drive downstream task transfer
(the ADR-0044 primary: COGS argument-role **+3.0**). Step 0 shows the abstract,
position-robust component of gold's advantage is **~1.5–2 points, not ~6.8** — so the
expectation that it carries a +3.0 COGS gain is *weakened, not supported*. The conclusion
is not "the effect is fake" (it is real and alignment-specific); it is "the invariant
signal is modest, so a full-corpus run should not be auto-launched expecting a downstream
win." A secondary objection — that the ~95% mixed-train passive ceiling *compresses* the
gold gap, hiding a larger true advantage — is plausible but unconfirmed; it would need a
harder, off-ceiling role readout to test, and is not support for proceeding on current
evidence.

## Decision (per the pre-registered rule) and options for the human

**Gate verdict: do NOT green-light Experiment 1 / cross the governance gate on this
evidence.** The strong crux failed; the invariant residual (+1.6) is below what the
downstream threshold (+3.0) needs. Reporting back before the ADR-0044 sign-off, as
required.

The human chooses among:
- **(a) Representational-only, redirect.** Treat the kāraka line as a confirmed
  *representational* effect that is mostly surface-tied, and stop here (a valid, honest
  closure state).
- **(b) Sharper invariant readout first.** Re-test the invariant component off the
  ceiling (harder/longer passives, dative role set) or jump to **Experiment 2
  (cross-lingual)** — the real invariance test (English-trained probe decoding Sanskrit
  roles); within-English voice transfer was only the warm-up.
- **(c) Proceed as a documented long-shot.** Run the full corpus accepting the invariant
  signal is ~1.6 pts, pre-registering that a COGS +3.0 is unlikely.

## Artifacts

- `scripts/probe_invariance.py`, `scripts/probe_invariance_mixed.py` (committed)
- `results/probe_invariance_results.json`, `results/probe_invariance_mixed_results.json`
- `results/probe_invariance.log`, `results/probe_invariance_mixed.log`
- Box `i-0a7d919aa390173a4` torn down; resume AMIs `ami-058f94b5b56335028` (known-good) +
  `ami-018040a752c6467d0` (post-Step-0) retained for Exp 1; stale AMI pruned.
