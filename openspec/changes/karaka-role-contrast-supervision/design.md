## Context

The program's distinctive lever — Pāṇinian structure — works cleanly only on
Sanskrit, where Vidyut *derives* morphology and roles by rule. On English (the
language actually trained and scored), the structure is recovered by generic
statistical tools (Morfessor for morphemes, spaCy for dependency roles), which
inject label noise. Three mechanisms (F2 masking, F3 auxiliary, F8 v0.2 masking)
returned null or dead, and the common root cause is **noisy English role labels
plus a broken injection point** (the per-vocab masking table cannot hold
per-instance roles; see the retired `english_deprel_mapper`).

Two facts make a different design possible:

1. **Kāraka is language-independent at the deep layer.** A giver/receiver/given
   structure survives translation even though every surface word changes
   (Kiparsky–Staal deep-vs-surface; cited in the paper). So gold roles minted on
   Sanskrit can be *transferred* to an English translation rather than guessed.
2. **Vibhakti decouples from kāraka, many-to-many.** One role surfaces through
   many cases (kartā → nominative in active, instrumental in passive, genitive
   with kṛt); one case marks many roles. This is normally a *problem* for parsing
   English; here it is an *asset* — it enumerates the construction alternations
   that hold a role invariant while varying the surface.

## Goals / Non-Goals

**Goals:**
- Produce **gold** (not estimated) per-token kāraka labels on English text.
- Teach **role-invariance** via contrast sets: one event, one role fixed, English
  surface varied (subject / by-phrase / possessive; to-dative / double-object).
- Inject the signal where it can actually reach the loss: the **supervised
  auxiliary role-prediction head** (the F3 path), not mask biasing.
- Target the **pre-registered** BLiMP agreement/argument-structure subset, so the
  test is honest and comparable to F2/F3.
- Make Vidyut load-bearing in the English path for the first time.

**Non-Goals:**
- Reviving mask-probability biasing or the `structured_masking` per-vocab table.
- Pañcāvayava / multi-sentence inference structure (that is H2, argument-level;
  a different corpus, unit, and exam).
- Transferring *vibhakti morphology* into English (English barely inflects;
  vibhakti is used only as the construction index, never as an English feature).
- Full-corpus gold labeling (gold parallel data is small; see Risks).
- Frontier parity or world-knowledge claims.

## Decisions

**D1 — Injection via the auxiliary head, not masking.** Roles enter as a
supervised token-level prediction target combined with MLM loss (the existing F3
mechanism with `RoleStreamPacker` alignment), discarded before evaluation. This
sidesteps the per-instance-vs-per-vocab impedance mismatch that killed the
masking path entirely: a parallel "role tape" aligned to tokens is exactly what
the aux head already consumes.

**D2 — Gold from the deep side, transferred by alignment.** Pipeline:
`seed event → Vidyut construction (gold kāraka known) → English translation →
word alignment → per-token gold label`. Alignment confidence gates the label;
below threshold a token is "unknown" (fall back), never a guessed role. We never
parse English for roles — the entire point is to *not* re-introduce spaCy noise.

**D3 — Vibhakti as the contrast index, English as the veto.** Enumerate
constructions per role from the case system; keep only cells that yield natural,
distinct English. Expected high-value cells: active↔passive (agent/patient),
dative alternation (recipient), instrument/source/locus prepositional frames.
Cells that collapse in English (e.g. instrument vs comitative vs means → "with")
are dropped and counted, not forced.

**D4 — Contrast as the unit of supervision.** The training unit is a *set* of
constructions sharing one invariant role, not a lone sentence. This is the
qualitatively-new signal English-only training cannot contain: the same
participant seen across surface frames with a constant gold label.

**D5 — Route fork (settled by gate 1.3: no attested parallel data exists
in-repo).** Two routes, trading opposite risks:

- **Route A — bilingual realization from frames (MVP).** Render the *same*
  `KarakaFrame` into both Sanskrit (existing `vidyut_realizer`) and English (a
  new, small frame→English realizer: ~46 glosses + active/passive/case-frame
  grammar). Roles are gold on **both** sides by construction, so **there is no
  translation and no word alignment** — the entire alignment risk dissolves. This
  is the purest form of the original idea (one meaning → two surfaces sharing one
  gold role set) and is fully buildable from repo assets. Cost: synthetic,
  templated, lexically narrow English. Kept as *auxiliary supervision* on a small
  set (not a pretraining corpus), which is the standard guard against the dose
  naturalness failure.
- **Route B — external parallel corpus (scale-up / external validity).** Ingest a
  real Sanskrit–English parallel corpus (e.g. Itihāsa, ~93K epic sentence pairs;
  or DCS-with-translations) for natural English at scale. Cost: a new data
  dependency, and roles must be projected onto English via parser + alignment —
  which partially reintroduces the very label noise we are escaping.

**Sequencing:** Route A first (cleanest possible isolation of the variable: gold
both sides, zero alignment confound → a null is interpretable, a win is real),
then Route B only if A shows signal (to test whether the effect survives natural
English rather than the template). Route A answers *internal* validity; Route B
answers *external* validity.

**D6 — Pre-register `H1_CONTRAST` with an explicit threshold and ≥3 seeds**,
paired bootstrap, Holm–Bonferroni, on the existing targeted subset — mirroring
F2/F3 so the result is directly comparable. Threshold value is set in the ADR
(default proposal: ≥ +1.0 targeted point, matching the short-paper pre-reg).

**D7 — Evaluation surface.** Primary candidates, both pre-registered:
(a) targeted BLiMP (agreement + argument-structure + passive/dative); (b) **COGS
argument-role categories** (`active_to_passive`, `passive_to_active`, the dative
alternations) — which the repo's own `cogs.py` calls "mechanism-aligned… where
the mechanism predicts" a benefit, and which match the contrast sets more
directly than BLiMP. Caveat: using the COGS *surface* does **not** reopen the
H1_COGS *dose* null (ADR-0017) — that was a different mechanism (synthetic
pre-pretraining), measured on the same benchmark. Secondary, exploratory: a
cross-lingual role-transfer probe — the program's untested cross-lingual angle,
not a closure gate.

## Risks / Trade-offs

- **Coverage vs. quality (central tension).** Aligned, role-annotatable
  Sanskrit↔English text is thousands–low-millions of words, not 100M. A small
  gold island may be underpowered — the same wall F3 hit. Mitigation: aux-head /
  fine-tune framing (not full-corpus), honest power analysis, distillation as a
  documented fallback. **Risk owner of the whole idea.**
- **Alignment risk — dissolves under Route A, dominant under Route B.** With
  bilingual realization (A) both surfaces come from one frame, so token→role
  attachment is gold by construction; no alignment step exists to break. The risk
  is real only for Route B (translation reordering, dropped pronouns,
  nominalization); there, mitigate with confidence-gated alignment and an
  "unknown" fallback rather than a forced label.
- **Synthetic-English naturalness — the dominant Route-A risk.** Frame-realized
  English is templated and narrow (≈38 stems / 8 verbs), so a *win* on A may be
  the model learning the template, not transferable role structure (the dose
  failure mode). Mitigations: keep it auxiliary supervision (not a corpus);
  grow the gloss lexicon; treat Route B as the external-validity check; and read
  a Route-A null as strong (perfect gold + zero confound still didn't move it).
- **Synthetic-text trap.** Combinatorially generated constructions can drift into
  stilted, templated language and teach the template (the dose failure mode).
  Mitigation: prefer attested sentences and natural alternations; cap
  combinatorial generation; treat vibhakti as a *checklist*, not a product space.
- **Vidyut generation: largely de-risked (gate 1.1).** Active + all six case
  frames already generate with test-verified gold role parses; passive is a
  contained wiring change; only kṛt is a real gap (deferred). This risk has
  moved *off* the Sanskrit side — the load now sits squarely on the
  English-transfer half (translation + alignment) and coverage, above.
- **Possibly still the wrong exam.** If the value is role-invariance, English
  BLiMP (surface grammar) may again under-detect it. Mitigation: the secondary
  cross-lingual probe; pre-commit to reporting both.
- **Null is an acceptable outcome.** Per the closure contract, a well-documented
  null (≥2 interventions) is a valid close. This design is built so that a null
  would be *informative* (gold labels + correct injection point ⇒ "the prior
  itself doesn't pay at this scale"), unlike the earlier nulls which were
  confounded by label noise or dead wiring.
