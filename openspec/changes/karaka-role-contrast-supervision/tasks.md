## 1. Feasibility gate — what can Vidyut actually generate? (RESOLVED — GREEN)

- [x] 1.1 Audited `vidyut_realizer.py` (the rich adapter). **Findings:**
      - **Already generated + test-verified:** active (kartari) finite verbs, and
        nominal declension in *all six* kāraka cases (nom/acc/instr/dat/abl/loc +
        genitive) via `Pada.Subanta(vibhakti=…)`. So instrument/recipient/source/
        locus role frames need **zero** new Sanskrit generation.
      - **Gold (surface, role) parse is real, not aspirational** — asserted by
        `test_realize_transitive_frame_gold_parse` (`has_gold_parse`). Validates
        design D2's premise outright.
      - **Passive (karmaṇi):** NOT wired — `Prayoga.Kartari` is hardcoded
        (`vidyut_realizer.py:339`) — but the `Prayoga` enum is API-reachable
        (`vidyut_source` parameterizes it). Adding it is contained, not research.
      - **Kṛt nominalization:** absent (no `Krdanta` calls anywhere). A larger lift.
      - **Dative alternation** is an English surface phenomenon; Sanskrit uses
        Caturthi (already generated). It emerges in translation, not generation.
- [x] 1.2 **Decision: route (a), Vidyut auto-generates.** MVP contrast set =
      active↔passive (highest value: hits BLiMP argument-structure/passive *and*
      COGS active↔passive) + the case-frame set (already done). Passive needs a
      voice-dependent kāraka→vibhakti map (kartā→Trtiya, karma→Prathama under
      Karmani) + threaded `prayoga`. **Kṛt deferred** to a stretch goal (lowest
      value, most stilted English). Curated fallback not needed for the MVP.
- [x] 1.3 Inventoried the repo's data assets. **Finding: attested parallel
      Sanskrit↔English in-repo = ZERO.** Wired corpora are Sanskrit-only
      (`sources.py`: GRETIL 449K verses, DCS morphology) or English-only (BabyLM);
      none are parallel. There is **no English realizer** and **no stem→English
      gloss table**. śābdabodha export is Sanskrit↔structured-*meaning*, not
      English. `data/corpora` and `data/synthetic` are empty; `data/nyaya_generated`
      (3.9M) is H2 reasoning data, not role pairs. The only gold asset is the
      *synthetic frame generator* (Vidyut realizer), Sanskrit-only and lexically
      narrow (≈38 stems / 8 verbs in the frame space). **Consequence:** the
      "small-clean *attested* parallel" assumption behind D5 does not hold from
      repo assets — forces the route fork now recorded in D5.

## 2. Pre-registration (governance)

- [x] 2.1 ADR drafted: `decisions/ADR-0043-h1-contrast.md` (canonical home
      `docs/decisions/`). Pre-registers `H1_CONTRAST`: hypothesis, **COGS
      argument-role subset as primary** (≥ +3.0 pts) with BLiMP targeted subset
      corroborating (≥ +1.0 pt), ≥3 seeds, paired bootstrap + Holm–Bonferroni,
      Route A scope. **Pending human sign-off** on framing + locked thresholds
      before any code (closure layer 6).
- [ ] 2.2 Define the secondary, exploratory cross-lingual role-transfer readout
      (probe + metric), explicitly non-gating.
- [x] 2.3 Matched baseline + shuffled-role specificity control specified in
      ADR-0043 (contrast is the only variable; `real − shuffled > 0` required).

## 3. Role-contrast set generator (domain + infrastructure)

- [x] 3.1 `construction_index.py` — the single source of truth for kāraka ↔
      vibhakti ↔ English (slot + preposition), incl. the passive voice shift
      (kartā nom→instr, karma acc→nom). `EnglishFrameRealizer` now sources every
      slot/preposition from it (no duplicate map). Lazy spaCy re-export in
      `linguistics/__init__.py` lets this pure module import without spaCy.
      12 tests in `test_construction_index.py`.
- [x] 3.2 `build_contrast_set(frame)` emits the constructions sharing a frame's
      gold role set (active + karmaṇi passive for transitive frames; active-only
      otherwise). Role-invariance asserted: active and passive share one noun-role
      set while the surface differs (`TestRoleInvariance`, `TestContrastSet`).
- [x] 3.3 English-veto = `english_collapses()` (groups constructions sharing one
      (slot, preposition)) + `log_coverage()` (per-voice coverage + drop warnings).
      **Honest finding:** at full-NP granularity the six kārakas each map to a
      distinct English device, so there is *nothing to drop* — coverage logs
      "6 kārakas realized, 0 collapse-group(s)". The drop-and-log mechanism is
      tested against a synthetic comitative-vs-instrument "with" collapse, so it
      fires when finer roles/pronouns are added later.
- [x] 3.4 Karmaṇi passive wired into `vidyut_realizer`: `voice` threads through
      `realize`/`_decline`/`_conjugate`; vibhakti is now voice-aware via the
      shared `construction_index` (`_vibhakti_for`), so kartā→Trtiya and
      karma→Prathama under Karmani; the verb takes Karmani prayoga and agrees with
      the patient. TDD against verified classical forms (naraH→nareRa,
      KAdati→KAdyate). Vidyut installed (0.4.0); 30 prior tests still green.

## 4. Gold English side (route fork per D5 / gate 1.3)

### Route A — frame→English realizer (MVP, no alignment)
- [x] 4.1 Gloss table authored + coverage-tested against the frame lexicon
      (14 nouns / 10 verbs incl. `xA1`, `vas1`) — `english_frame_realizer.py`
      (`NOUN_GLOSS`, `VERB_GLOSS`); `test_english_frame_realizer.py::TestGlossCoverage`.
- [x] 4.2 `EnglishFrameRealizer.realize(frame, voice=...)` renders **active** and
      **karmaṇi passive**, plus all oblique case-frames (instrument "with" /
      recipient "to" / source "from" / locus "in"), with number + tense agreement
      (passive aux agrees with the patient) and a gold `karaka_parse` (WX role
      vocabulary, pure — no parser). 23 tests green, ruff + mypy clean.
- [x] 4.3 SA↔EN cross-check done (`test_bilingual_contrast.py`): one frame
      realized by both realizers shares one gold role inventory, and that
      inventory is invariant across language × voice (active/passive, over a
      50-frame stream). Surfaces differ entirely (naraH/nareRa vs "the man"/"by
      the man") while {karwA, karma, karaNam, kriyā} is constant.

### Route B — external parallel + alignment (scale-up, deferred until A shows signal)
- [ ] 4.4 Ingest an external Sanskrit–English parallel corpus (e.g. Itihāsa); TDD
      a word-alignment adapter behind a port, gated like the optional `vidyut`.
- [ ] 4.5 TDD: transfer module — gold Sanskrit roles → English tokens via
      alignment; confidence threshold; sub-threshold tokens → `unknown`.
- [ ] 4.6 Regression guard (both routes): assert no English role parser is ever
      called — no spaCy-derived roles enter the label stream.

## 5. Supervision wiring (reuse F3 aux-head path)

Scoping found the training side is already generic — the F3 aux head, loss, and
`RoleStreamPacker` consume a precomputed `uint8` role `.bin`; no model/loss/loop
changes are needed. Work is on the *data* side: produce a **gold** role stream.

- [x] 5.0 `EnglishFrameRealizer.word_roles(frame, voice)` — per-surface-word
      gold roles (head→kāraka, function words→`separator`), via the shared
      `_chunks`; realize() refactored onto it (27 tests).
- [x] 5.1 `contrast_corpus.py`: `WX_TO_SHABDABODHA` + `wx_role_id`,
      `sentence_role_ids` (reuses `align_pieces_to_role_ids`), `build_contrast_corpus`
      (lines + flat role-id stream, EOS role, alignment invariant
      `len(roles)==tokens+lines`), and `shuffle_roles` (ADR shuffled-role control).
- [x] 5.2 Confound-removal **measured**: on the committed `spm.model`, every gold
      kāraka lands on its head piece (`TestRealTokenizerAlignment`) — ~0% residual,
      unlike the spaCy path. No `structured_masking` import anywhere (mask path
      untouched, by construction — the dead F8 socket stays retired).
- [x] 5.3 Aux-wiring contract test (`test_contrast_aux_wiring.py`): gold stream →
      `RoleStreamPacker` → `ShabdabodhaHead` → `shabdabodha_aux_loss` gives a
      finite loss that responds to the labels. Lazy-imported spaCy out of
      `shabdabodha_target` so the label set imports without it.
- [x] 5.4 Mixed-corpus assembly + cache writer. `contrast_corpus.py` adds
      `corpus_with_roles` / `none_role_lines` / `mix_lines` / `flatten_corpus`
      (Option B: gold roles on contrast tokens, `none` on background, deterministic
      interleave). `scripts/build_contrast_cache.py` writes the trainer-ready cache
      (`english_base.txt`, `english_base.bin` uint16, `shabdabodha_roles_eos.bin`
      uint8, + shuffled control) with the hard alignment check; smoke-run OK
      (510 lines / 5049 tokens / 5559 roles, invariant holds). No
      `train_submission_model` change — its existing `--shabdabodha-aux λ
      --shabdabodha-roles <gold>.bin` flags consume the cache directly.

## 6. Experiments + closure

- [ ] 6.1 Run the matched baseline and the `H1_CONTRAST` arm at proxy scale; log
      coverage (tokens attempted/retained/dropped) per the honesty requirement.
- [ ] 6.2 Compute targeted-subset contrast + 95% CI vs. threshold; run the
      shuffled-role specificity control (rule out generic multi-task gain).
- [ ] 6.3 If threshold missed: run ≥2 documented interventions (alignment quality,
      coverage/distillation, construction mix) before any null is declared.
- [ ] 6.4 Report the secondary cross-lingual probe alongside the primary result.
- [ ] 6.5 Write the finding (positive / marginal / null) + interpretation +
      Tarka memo (strongest objection to your own result) to the experiment
      ledger; update the paper section from the finding.
