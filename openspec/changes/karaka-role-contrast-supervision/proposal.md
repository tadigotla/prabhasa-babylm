## Why

Every Pāṇinian *masking* mechanism tried so far has come back null, and the
diagnosis is consistent across the program:

- **F2 (masking, paper):** at matched mask budget, kāraka-stratified masking is
  causally inert (Δ = +0.10, ns).
- **F3 (auxiliary objective, paper):** the role-prediction head gives only +0.76
  targeted points over five seeds (ns), attenuating with each added seed.
- **F8 (v0.2):** the M2 masking mechanisms are dead — `english_deprel_mapper`
  produces real per-instance roles but its probability methods return a flat
  default, so the deprel/MI arms ran *byte-identical to control* ("NOT wired →
  no-ops, untested"). The masking line was retired.

Underneath all three sits the limitation the paper itself names first: **role
labels on English are guessed, not derived.** spaCy/Morfessor estimate English
structure statistically and inject *label noise*; the paper flags "replacing it
with dependency-parsed roles" as "the most direct intervention for F2 and F3."

This change takes that intervention to its clean limit. Instead of *guessing*
English roles, we **mint them as gold on the Sanskrit side — where Vidyut derives
them by rule — and transfer them to English through aligned translation.** The
roles are gold by construction (we build the sentence knowing its kāraka), and
they survive translation because kāraka is a semantic, language-independent layer
(the deep/surface decoupling of Kiparsky–Staal that the paper already cites).

The signal is sharpened by **role-contrast**: Sanskrit's construction
alternations (active ↔ passive ↔ nominal, and the dative alternation) re-express
*one event* with *one role held invariant* while the *surface* varies. The
vibhakti (case) system is the complete index of which alternations exist, so the
contrast set is systematic rather than ad hoc. The model is taught the one thing
the program is about: **a role is the participant's relation to the event, not
the word's identity.**

This is also the first design in which Vidyut earns its place in the *English*
path (as the labeling oracle) rather than being switched off (`vidyut_available=
False` in the submission run).

## What Changes

- **A new H1 mechanism, pre-registered as `H1_CONTRAST`:** gold kāraka
  role-contrast supervision transferred from Sanskrit to English, injected as a
  *supervised auxiliary role-prediction signal* (reusing the F3 aux-head path),
  **not** as mask-probability biasing (the retired, dead socket).
- **A role-contrast data generator:** from a seed event, emit a *set* of
  constructions (active / passive / nominalization / dative alternation) that
  hold one kāraka invariant while varying the English surface. Vibhakti indexes
  the construction menu; English vetoes cells it cannot express naturally.
- **A gold-label English side.** Gate 1.3 found *no* attested Sanskrit↔English
  parallel data in-repo, so the MVP (**Route A**) realizes the *same* kāraka
  frame into both Sanskrit and English — roles gold on both sides by construction,
  **no translation or alignment**. **Route B** (external parallel corpus +
  confidence-gated alignment, falling back to "unknown", never a guessed role) is
  the natural-language scale-up once Route A shows signal. See design D5.
- **Evaluation against the existing pre-registered targeted subset** (BLiMP
  agreement + argument-structure, plus passive/dative paradigms), with a
  secondary, exploratory cross-lingual role-transfer readout.
- **Honesty instrumentation:** coverage and dropped-cell counts are logged
  (gold parallel data is small; silent truncation is forbidden).

This change does **not** revive mask-biasing, does **not** use pañcāvayava /
inference structure (that is H2, argument-level, out of scope here), and does
**not** claim frontier parity.

## Capabilities

### New Capabilities
- `role-contrast-supervision`: generation of gold kāraka role-contrast sets from
  Sanskrit, transfer of those gold labels to English via aligned translation, and
  their injection as a supervised auxiliary role-prediction signal during English
  pretraining, evaluated against a pre-registered targeted BLiMP subset.

### Modified Capabilities
<!-- No existing openspec/specs/ capabilities yet; this is the first spec. -->

## Impact

- **Hypotheses / governance:** introduces `H1_CONTRAST`; pre-registering its
  threshold requires an ADR in `docs/decisions/` (per CLAUDE.md). Sits beside the
  frozen masking arms; does not alter them.
- **Code (infrastructure / application layers, hexagonal):**
  - `infrastructure/generators/` — `vidyut_realizer` already emits active forms
    and all six kāraka case frames with a *test-verified gold role parse*; the
    only generation work for the MVP is wiring karmaṇi (passive) voice (contained;
    see gate 1.1). Kṛt nominalization is deferred.
  - new `infrastructure/linguistics/` (or `domain/linguistics/`) — role-contrast
    set construction; vibhakti→construction index.
  - new `infrastructure/ml/` transfer module — translation + word alignment →
    gold per-token role labels with a confidence gate.
  - reuse the F3 auxiliary-head + `RoleStreamPacker` path (`elc_trainer`,
    `train_submission_model`); **no** changes to `structured_masking` mask probs.
- **Dependencies:** a word aligner (e.g. SimAlign/awesome-align or statistical
  alignment) and a Sanskrit→English MT or curated parallel source; gated as
  optional extras, consistent with the optional-`vidyut` pattern.
- **Feasibility gate (task 1): RESOLVED — green.** `vidyut_realizer` already
  generates active + all six case frames with gold, test-verified role parses;
  passive is a contained wiring change; kṛt is deferred. Route (a) (Vidyut
  auto-generates) is adopted. The real remaining risk is the English-transfer
  half (translation + alignment) and coverage, not Sanskrit generation.
- **Data provenance:** corpus, configs, and the resolved config hash recorded per
  run, as for every PSALM experiment.
