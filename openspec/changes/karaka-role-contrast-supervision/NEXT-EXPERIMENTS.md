# H1_CONTRAST — next experiments (triggerable plan)

Hand this to a fresh session. Self-contained: box bring-up, exact configs, the
sequenced experiments, the **metric framework (see ADR-0044)**, thresholds,
analysis, and decision rules.

## Where we are (one paragraph)

Mechanism built confound-free (gold kāraka roles from Sanskrit + frame-realized
English, injected via the F3 śābdabodha aux head, with a shuffled-role specificity
control). At **proxy scale (1.3M-token sample) BLiMP read null** — a *wrong-
instrument* artifact (BLiMP = surface grammar; mechanism = semantic roles). A
**linear argument-role probe (n=9)** showed gold makes argument roles
**significantly more decodable** from the frozen encoder, generalizing to held-out
natural English: **gold−baseline +6.8 (t=5.6, 9/9 seeds); gold−shuffled +3.8
(t=3.0, alignment-specific)**. The plan below was sharpened by a design review
(folded in: §"Metric framework" and the per-experiment notes).

---

## Metric framework (the key sharpening — pre-registered in ADR-0044)

The proxy run substituted BLiMP for the ADR-0043 locked COGS primary because COGS
was not computable as-is. The probe then found the effect BLiMP couldn't see.
**Before the full run, ADR-0044 re-designates the metrics** (pre-registration, not
post-hoc):

| Role | Metric | Why |
|---|---|---|
| **Primary (task)** | **COGS argument-role discrimination**, via a *to-be-built* COGS-disc-for-MLM scorer | Honors the ADR-0043 locked primary; a *role-dependent* task (roles needed to build the LF), not BLiMP-blind, not fine-tune-circular |
| **Confirmatory + specificity** | **Linear role probe, n ≥ 9** (the powered instrument) | Carries the gold−shuffled specificity test — the probe needed n=9 to clear zero; BLiMP-PLL can't at n=3 |
| **Corroborating (coarse)** | **BLiMP role-sensitive subset** (arg-structure / passive / causative / arg-drop / inchoative — **agreement EXCLUDED**) | Floor-lift + direction only; NOT the specificity gate; agreement is morphosyntax, a role-irrelevant confound |
| **Crux pre-check** | **Construction-invariance probe** (train-active → test-passive) | Is the role subspace invariant to a surface change? Necessary condition for downstream *and* cross-lingual |

Honest caveat to inherit: for a method evaluated through an *MLM encoder*,
"downstream task accuracy" is inherently **probe-flavored** (linear probe → head →
fine-tune is a capacity continuum; the MLM doesn't *do* a task). The COGS-disc-for-
MLM scorer is a *structured* probe — the most task-shaped, pre-registered one. A
fine-tune that predicts roles is just a higher-capacity probe (circular); the
non-circular task is one where roles *help* but aren't the label, which COGS is.

---

## Shared session kickoff

**Box.** Relaunch the resume AMI that already contains everything from the probe
session — code in `~/PSALM`, the **27 checkpoints** under `runs/contrast_matrix/`,
the 1.3M contrast cache, the probe/matrix scripts, and the COGS+BLiMP eval data
cached. Use the **latest `project=psalm` resume AMI** (at session close it was
`ami-058f94b5b56335028`; resolve the newest in case it was re-baked/pruned).

```bash
AMI=$(aws ec2 describe-images --region us-east-1 --owners self \
  --filters Name=tag:project,Values=psalm \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
TTL=$(date -u -v+8H +%Y-%m-%dT%H:%M:%SZ)   # macOS; Linux: date -u -d '+8 hours'
cap --region us-east-1 launch psalm --type g6.xlarge --ami "$AMI" --key psalm-train \
  --security-group sg-0bd4bd7ced442b676 --subnet subnet-844e62ce --disk 100 \
  --expires "$TTL" --yes
MYIP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --region us-east-1 \
  --group-id sg-0bd4bd7ced442b676 --protocol tcp --port 22 --cidr "$MYIP/32" || true
IP=$(aws ec2 describe-instances --region us-east-1 \
  --filters Name=tag:project,Values=psalm Name=instance-state-name,Values=running,pending \
  --query 'Reservations[].Instances[].PublicIpAddress' --output text)
ssh-keyscan -H "$IP" >> ~/.ssh/known_hosts 2>/dev/null
ssh -i ~/.ssh/psalm-train.pem ubuntu@"$IP"
cd ~/PSALM && export PATH="$HOME/.local/bin:$PATH" && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
Lifecycle (launch/TTL/teardown/prune) in
`/Users/adigo/code/2026/assets/aws-mgmt/docs/psalm-gpu-runbook.md`. **Tear down
when done:** `cap --region us-east-1 end psalm`; prune old AMIs to the newest.

**Measured config gotchas (do not rediscover):**
- Tokenizer **vocab 8000** → always `--vocab 8000`.
- Default `--batch-size 256` **OOMs the 24 GB L4** → use `--batch-size 32`
  (≈40K tok/s, ~3.8 min per 7-epoch run on 1.3M tokens).
- Default `--dose-arms` reads `A B C D` but only `A`,`B` exist → `--dose-arms A
  --dose-epochs 0`.
- Clean isolation (only aux differs): `--no-nhot-embeddings --no-structured-masking
  --freq-alpha 0 --no-babylm-checkpoints`.
- Arms: baseline `--shabdabodha-aux 0`; gold `--shabdabodha-aux 1.0
  --shabdabodha-roles <dir>/shabdabodha_roles_eos.bin`; shuffled → `*_shuffled.bin`.
- On the box: `~/PSALM/probe_roles.py`, `run_matrix.sh`, `run_seeds_3to8.sh`. In
  repo: `scripts/build_contrast_cache.py`, `scripts/probe_argument_roles.py`.
- Branch is on `github.com/tadigotla/prabhasa-babylm`
  (`feature/karaka-role-contrast-supervision`).

---

## Step 0 — Construction-invariance probe (DO FIRST; ~15 min; existing checkpoints)

**Highest information-per-GPU-hour move, and nearly free.** Tests the crux: is the
gold role subspace invariant to a surface change (active↔passive)?

- Clone `probe_roles.py` → `probe_invariance.py`. Same rep extraction, but **train
  the probe on ACTIVE sentences only, test on PASSIVE** (same roles, flipped
  surface). Sources: COGS `active_to_passive` / `passive_to_active` gen categories,
  or our gold contrast active/passive realizations (`build_contrast_corpus` emits
  both, with gold roles).
- Run over the existing 27 checkpoints (9 seeds × 3 arms). Compare gold vs baseline
  vs shuffled on **train-active → test-passive** accuracy.
- **Decision:** gold's active→passive transfer significantly > baseline → the role
  subspace is construction-invariant (necessary for both downstream and cross-
  lingual) → green-light Experiment 1 and 2. If it collapses to baseline, the
  +6.8 was surface-tied — reconsider before spending GPU-hours.

---

## Governance gate — sign off ADR-0044 BEFORE the full run

`decisions/ADR-0044-h1-contrast-metrics.md` re-designates the primary metric
(COGS task) and the role of the probe/BLiMP. **It must be signed off before
looking at any full-run results**, or the metric choice is post-hoc. (Per CLAUDE.md,
changing a pre-registered metric needs an ADR.)

---

## Experiment 1 — Full-corpus downstream-translation test

**Question.** Does the representational gain translate into **task accuracy** once
the model trains on the full 10M strict-small corpus (metrics off the chance floor)?

**Metrics:** per the framework above (COGS primary; probe confirmatory+specificity;
BLiMP corroborating with agreement dropped). Pre-registered in ADR-0044.

**Steps.**
1. **Build the COGS-disc-for-MLM scorer** (the open recommendation #2 — this is the
   primary metric, build it before the run). Structured readout over the frozen
   encoder for COGS argument-role gen; reuse `cogs.load_cogs`,
   `domain.eval.discrimination.CORRUPTIONS`, and the `load_elc_checkpoint` /
   `hidden_mlm` extraction from `probe_roles.py`. Honest: it is a structured probe.
2. **Materialize the full 10M corpus** → `data/corpora/strict_full/english_base.txt`.
   **Reuse `scripts/assemble_strict_small.py` / `scripts/prepare_babylm_100m.py`
   if present** before writing new prep. Verify ~10M words / token count.
3. **Build the mixed gold cache — MAX diversity, no repetition.** The generator
   ceiling is **~74,760 unique frames → ~125k EN lines → ~1.24M contrast tokens**
   (measured). Use `--n-frames 74760` (or the realized max) with **`--contrast-repeat
   1`** — same ~1.2M-token budget as repeat-7 but ~7.5× more unique surfaces (serves
   D4). At 10M BabyLM that is ~11% contrast tokens — already in-band. Only add a
   small top-up repeat (≤1.5×) if you insist on 15%.
   - **Lexicon caveat:** even 125k lines recombine only ~24 content words (14 nouns /
     10 verbs); the diversity gain is structure/voice/number/tense, NOT vocabulary.
     The deeper fix (future) is expanding the lexicon via the realizer's DCS-grounded
     native-frame path. **Log unique-line count AND unique-content-word count**
     alongside the token fraction and the alignment invariant.
4. **Train the matrix:** 3 arms × **3 seeds** (more if budget), full corpus, clean
   config. **Batch: grad-accum to effective 256 — mandatory, not optional** (lifts
   metrics off the floor AND stays matched across arms; needs a `--grad-accum` flag
   if absent). Per-run longer than proxy (10M×7ep); budget ~30–60 min/run.
5. **Eval each checkpoint:** COGS-disc scorer (primary), `probe_roles.py` at the
   full-corpus checkpoints (confirmatory + specificity), `eval_blimp_pll.py
   --per-paradigm 200` over the role-sensitive subset only (corroborating).
6. **Analyze & decide:**
   - **COGS gold−baseline ≥ +3.0 AND probe gold−shuffled significant** → positive
     downstream translation; write up.
   - **Probe holds but COGS/BLiMP flat** → "representational, not (yet) task" —
     honest; motivates a different task surface or Experiment 2.
   - **All flat at full scale** → the proxy probe effect doesn't scale; reconsider.

**Risks:** corpus-prep correctness (reuse + verify counts/alignment); COGS-for-MLM is
itself probe-flavored (state it); strict-small BLiMP ceilings are modest (~65–72) so
the *gap* matters more than the level.

---

## Experiment 2 — Cross-lingual role transfer (the language-independence test)

**Deep question.** Are kāraka roles encoded language-independently — does a probe
trained on one language decode another's roles? (Step 0 is the within-English
warmup; this is the real test.) Current 27 checkpoints are **English-only** and
cannot represent Sanskrit tokens, so 2-full needs a **bilingual model first.**

1. **Tokenizer.** Train a new SentencePiece tokenizer on **EN + SA** (the 8000
   English vocab doesn't cover SLP1 Sanskrit). `SentencePieceTrainer` is in the repo.
2. **Bilingual gold corpus.** Generate SA contrast realizations
   (`VidyutFrameRealizer`, gold `karaka_parse`) + EN contrast + BabyLM; **extend
   `contrast_corpus.py`** to emit SA lines + SA per-token role ids (SA `karaka_parse`
   → per-token roles via the new tokenizer; reuse `align_pieces_to_role_ids`). Aux
   trains role prediction on both languages.
3. **Train** the bilingual baseline/gold/shuffled matrix (≥3 seeds).
4. **Cross-lingual probe.** Train the role probe on **English** reps, test on
   **Sanskrit** reps (and vice versa). **Gold's English-trained probe decoding
   Sanskrit roles above baseline/shuffled = language-independent role encoding** —
   the direct evidence for the Pāṇinian claim, and the paper's centerpiece.

**Time/cost:** bigger build — bilingual tokenizer (~30 min) + bilingual cache (new
code, ~half a day dev) + training (~5 h) + probe. Plan 1–2 sessions. Pursue if
Step 0 + Exp 1 are encouraging.

---

## Recommended order

1. **Step 0** (construction-invariance) — ~15 min, existing checkpoints, tests the
   crux abstraction claim. Do this first.
2. **Sign off ADR-0044** (metric re-designation) before any full run.
3. **Experiment 1** (full-corpus) — with the COGS-disc scorer as primary, probe
   carrying specificity, BLiMP corroborating (agreement dropped), max-diversity
   corpus, grad-accum-256.
4. **Experiment 2** (bilingual cross-lingual) — the deepest result and biggest
   build; pursue if 1 + Step 0 are encouraging.

Honest framing throughout: report positive / representational-only / null as it
lands, with a Tarka memo (strongest objection to your own result), per the closure
contract.
