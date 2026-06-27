# H1_CONTRAST — task 6 empirical pass (proxy scale, L4)

**Date:** 2026-06-27 · **Hardware:** NVIDIA L4 (23 GB), us-east-1 · **Status:** documented (preliminary, underpowered)

## What ran

The full H1_CONTRAST pipeline executed end-to-end on GPU for the first time:
gold kāraka role-contrast supervision (Route A bilingual realization) → mixed
corpus (16.7k contrast + 87k BabyLM lines, gold roles on contrast tokens, `none`
on BabyLM) → śābdabodha auxiliary head → 3 arms × 3 seeds, pure MLM + aux,
extraneous mechanisms (N-hot, structured masking, freq) OFF so the **only**
variable across arms is the role stream.

- **baseline**: aux off (pure MLM)
- **gold**: aux on, gold-aligned roles
- **shuffled**: aux on, role multiset preserved, token alignment destroyed (the
  ADR-0043 specificity control)

Metric: BLiMP pseudo-log-likelihood minimal-pair accuracy (the pre-registered
COGS argument-role primary was **not computable** — the repo's COGS discrimination
eval trains its own causal models; it is not a scorer for an MLM checkpoint).

## Numbers (3 seeds)

```
BLiMP OVERALL (67 paradigms)      baseline 51.74   gold 52.83   shuffled 51.49
  gold − baseline  +1.09 pts  95% CI [-0.59, +2.77]  t=+2.79  (seeds +1.87,+0.78,+0.63)
  gold − shuffled  +1.34 pts  95% CI [-2.19, +4.87]  t=+1.63  (seeds -0.13,+1.45,+2.70)
  shuffled − base   -0.25 pts  (NS)

BLiMP TARGETED (15: arg-structure / passive / subject-verb agreement)
                                  baseline 56.64   gold 56.22   shuffled 55.56
  gold − baseline  -0.42 pts  95% CI [-1.60, +0.75]  t=-1.55
  gold − shuffled  +0.67 pts  95% CI [-1.88, +3.22]  t=+1.13
```

## Finding: NULL on the mechanism-aligned readout (preliminary, underpowered)

The small overall gold−baseline edge (+1.09, all three seeds positive) **does not
appear on the targeted argument-structure/passive/agreement paradigms** (−0.42),
which is exactly where the kāraka role-contrast mechanism predicts it should. The
gold−shuffled specificity contrast is +0.67 targeted / +1.34 overall, neither
significant at n=3. The pre-registered thresholds (COGS +3.0; BLiMP targeted +1.0)
are **not met**.

So at this proxy scale the gold role-contrast signal gives **no role-specific
benefit on the paradigms it targets** — consistent with the program's prior
F2/F3 kāraka nulls.

## Tarka memo (strongest objection to the null)

One pattern resists the null: on BLiMP overall, gold beats baseline in **all three
seeds** and beats the shuffled control while **shuffled ≈ baseline** — the
signature of a real, alignment-specific effect rather than generic multi-task
regularization. The targeted-subset null could itself be an artifact of (a) the
ad-hoc 15-paradigm subset (vs the pre-registered 20), (b) near-chance saturation —
absolute accuracy is ~52% overall / ~56% targeted, the model barely off the floor,
(c) the corpus being a **~1.3M-token sample** (≈1/8 of strict-small), giving the
aux almost nothing to latch onto. So this is **weak evidence at proxy scale, not a
refutation.**

## Honest limitations

- **Corpus is a 1.3M-token sample**, not the 10M strict-small budget → BLiMP sits
  at chance, the regime with least signal.
- **n=3 seeds** → CIs span ±3–5 points; badly underpowered.
- **Primary metric (COGS) not computed** — BLiMP is a corroborating proxy.
- **Batch 32** (L4 memory) ≠ the recipe's effective batch 256.

## Recommendation (next interventions, per closure contract — ≥2 before any NULL)

1. **Full-corpus rerun** at the 10M strict-small budget so BLiMP lifts off the
   floor (the proxy regime is uninformative). #1 priority.
2. **Build a COGS-discrimination-for-MLM scorer** (teacher-forced role-corrupted-LF
   PLL on the encoder) so the pre-registered primary metric is actually measured.
3. **≥5 seeds** + recipe-faithful effective batch (grad-accum).

The deliverable that *is* solid: H1_CONTRAST is now a runnable, confound-free
experiment (gold labels, correct injection point, specificity control) — the
pipeline the earlier nulls never had.
