## ADDED Requirements

### Requirement: Gold role-contrast set generation
The system SHALL generate, from a seed event, a *set* of Sanskrit constructions
that hold one kāraka role invariant while varying the surface realization, with
each token's kāraka role known by construction (gold, not estimated). The
construction menu SHALL be enumerated from the vibhakti (case) system.

#### Scenario: One role held invariant across constructions
- **WHEN** a seed event with a designated agent (kartā) is expanded into a
  contrast set
- **THEN** the set includes at least the active (kartari) and passive (karmaṇi)
  constructions, and in every member the agent token carries the gold label
  `karta` despite differing surface case/position

#### Scenario: Gold labels carry provenance
- **WHEN** any sentence in a contrast set is emitted
- **THEN** each token's kāraka label is tagged as `gold` with the deriving
  construction recorded, and no token role is produced by parsing surface text

### Requirement: Gold kāraka labels on the English side, never parser-guessed
The system SHALL produce per-token gold kāraka labels on the English side, and
SHALL NOT assign any English token a role inferred from an English parser. The
labels MAY be produced either by co-realizing the same kāraka frame into English
(Route A: gold by construction, no alignment) or by confidence-gated word
alignment to a gold-labeled Sanskrit token (Route B), per design D5.

#### Scenario: Route A — English co-realized from the frame is gold by construction
- **WHEN** a kāraka frame is realized into both Sanskrit and English
- **THEN** each English content token carries the frame's gold kāraka role with no
  alignment step and no English-parser inference

#### Scenario: Route B — high-confidence alignment yields a gold English label
- **WHEN** Route B is used and an English token aligns to a gold-labeled Sanskrit
  token with confidence at or above the threshold
- **THEN** the English token receives that gold kāraka label

#### Scenario: Route B — low-confidence alignment falls back, never guesses
- **WHEN** Route B is used and an English token's best alignment confidence is
  below the threshold
- **THEN** the English token is labeled `unknown` (default fallback) and is never
  assigned a role inferred from an English parser

### Requirement: Vibhakti-indexed coverage with English veto and drop logging
The system SHALL treat the vibhakti system as the index of candidate
constructions, retain only constructions that yield natural, distinct English,
and SHALL record every dropped construction with a reason. The system SHALL NOT
transfer vibhakti morphology as an English feature.

#### Scenario: Untranslatable case distinction is dropped, not forced
- **WHEN** two Sanskrit constructions differ only by a case distinction English
  collapses (e.g. instrument vs. comitative, both surfacing as "with")
- **THEN** the redundant construction is dropped and the drop is logged with the
  collapsing-distinction reason

#### Scenario: Coverage is reported, never silently truncated
- **WHEN** a contrast-set generation run completes
- **THEN** the run logs the count of constructions attempted, retained, and
  dropped per role, so coverage limits are visible in the run record

### Requirement: Supervision via the auxiliary role-prediction head
The system SHALL inject role-contrast supervision as a token-level supervised
auxiliary objective combined with the MLM loss, and SHALL NOT modify
mask-selection probabilities to carry the role signal. The auxiliary head SHALL
be discarded before evaluation.

#### Scenario: Roles drive an auxiliary loss, not the mask distribution
- **WHEN** training runs with role-contrast supervision enabled
- **THEN** the per-token gold roles feed the auxiliary role-prediction loss, the
  mask-selection distribution is unchanged from the matched baseline, and no role
  labels are required at evaluation time

#### Scenario: Disabled flag is a true no-op
- **WHEN** role-contrast supervision is disabled
- **THEN** training is byte-identical to the matched baseline (regression guard
  against the F8 "looks-wired-but-inert" failure)

### Requirement: Pre-registered evaluation and null handling
The system SHALL evaluate `H1_CONTRAST` against the pre-registered targeted BLiMP
subset across at least three seeds with paired bootstrap inference and
Holm–Bonferroni correction, report a finding (positive / marginal / null) with an
interpretation, and treat a documented null as a valid outcome.

#### Scenario: Result is computed against the pre-registered threshold
- **WHEN** the seeds for an arm complete
- **THEN** the targeted-subset contrast versus the matched baseline is computed
  with its 95% CI and compared to the pre-registered `H1_CONTRAST` threshold, and
  the finding plus interpretation are written to the experiment ledger

#### Scenario: A null is closed honestly, not declared on attempt one
- **WHEN** the first arm misses the threshold
- **THEN** at least two documented interventions are run before any null is
  declared, each logged with attempt number, change, and result, per the closure
  contract
