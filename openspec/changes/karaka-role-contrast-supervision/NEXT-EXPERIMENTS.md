# H1_CONTRAST — next experiments (triggerable plan)

Hand this to a fresh session. It is self-contained: box bring-up, exact configs,
the two experiments, thresholds, analysis, and decision rules.

## Where we are (one paragraph)

The mechanism is built confound-free (gold kāraka roles from Sanskrit + frame-
realized English, injected via the F3 śābdabodha aux head, with a shuffled-role
specificity control). At **proxy scale (1.3M-token sample), BLiMP read null** — but
that was a *wrong-instrument* artifact: BLiMP tests surface grammar, the mechanism
injects semantic roles. A **linear argument-role probe (n=9)** showed gold makes
argument roles **significantly more decodable** from the frozen encoder,
generalizing to held-out natural English: **gold−baseline +6.8 (t=5.6, 9/9 seeds);
gold−shuffled +3.8 (t=3.0, alignment-specific)**. Two open questions follow.

---

## Shared session kickoff (both experiments)

**Box.** Relaunch the resume AMI that already contains everything from the probe
session — my code in `~/PSALM`, the 27 trained checkpoints under
`runs/contrast_matrix/`, the 1.3M contrast cache, the probe/matrix scripts, and the
COGS+BLiMP eval data cached. Use the **latest `project=psalm` resume AMI**
(at session close it was `ami-058f94b5b56335028`; resolve the newest in case it was
re-baked or pruned).

```bash
AMI=$(aws ec2 describe-images --region us-east-1 --owners self \
  --filters Name=tag:project,Values=psalm \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
TTL=$(date -u -v+8H +%Y-%m-%dT%H:%M:%SZ)   # macOS; Linux: date -u -d '+8 hours'
cap --region us-east-1 launch psalm --type g6.xlarge --ami "$AMI" --key psalm-train \
  --security-group sg-0bd4bd7ced442b676 --subnet subnet-844e62ce --disk 100 \
  --expires "$TTL" --yes
# allowlist THIS machine's IP (changes per session):
MYIP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --region us-east-1 \
  --group-id sg-0bd4bd7ced442b676 --protocol tcp --port 22 --cidr "$MYIP/32" || true
IP=$(aws ec2 describe-instances --region us-east-1 \
  --filters Name=tag:project,Values=psalm Name=instance-state-name,Values=running,pending \
  --query 'Reservations[].Instances[].PublicIpAddress' --output text)
ssh-keyscan -H "$IP" >> ~/.ssh/known_hosts 2>/dev/null
ssh -i ~/.ssh/psalm-train.pem ubuntu@"$IP"
# on the box:
cd ~/PSALM && export PATH="$HOME/.local/bin:$PATH" && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
Full lifecycle (launch/TTL/teardown/prune) is in
`/Users/adigo/code/2026/assets/aws-mgmt/docs/psalm-gpu-runbook.md`. **Tear down
when done:** `cap --region us-east-1 end psalm` (bakes a fresh resume AMI, stops
billing); prune old AMIs to the newest per the runbook.

**Measured config gotchas (do not rediscover these):**
- Tokenizer is **vocab 8000** → always pass `--vocab 8000`.
- Default `--batch-size 256` **OOMs the 24 GB L4** with the routing arch → use
  `--batch-size 32` (≈40K tok/s, ~3.8 min per 7-epoch run on 1.3M tokens).
- Default `--dose-arms` reads `A B C D` but only `A`,`B` exist → pass
  `--dose-arms A --dose-epochs 0` (the 1 forced dose step is negligible, matched
  across arms).
- Clean-isolation flags (only the aux differs across arms):
  `--no-nhot-embeddings --no-structured-masking --freq-alpha 0 --no-babylm-checkpoints`.
- Arms: baseline `--shabdabodha-aux 0`; gold `--shabdabodha-aux 1.0
  --shabdabodha-roles <dir>/shabdabodha_roles_eos.bin`; shuffled → the
  `*_shuffled.bin`.
- Reusable scripts already on the box: `~/PSALM/probe_roles.py`,
  `run_matrix.sh`, `run_seeds_3to8.sh`. In the repo: `scripts/build_contrast_cache.py`,
  `scripts/probe_argument_roles.py`.

If you relaunch a CLEAN AMI instead, re-overlay code: the branch is on
`github.com/tadigotla/prabhasa-babylm` (`feature/karaka-role-contrast-supervision`);
overlay the changed files into `~/PSALM` (tar-over-ssh or git-archive), since the
box `~/PSALM` is a non-git working tree with a baked venv.

---

## Experiment 1 — Full-corpus downstream-translation test

**Question.** Does the representational gain (proven on the proxy) translate into
**downstream task accuracy** once the model trains on the full 10M strict-small
corpus, so BLiMP/COGS lift off the chance floor?

**Pre-register at kickoff (lock before looking):**
- Primary: **BLiMP targeted subset** (arg-structure/passive/agreement) gold−baseline
  **≥ +1.0** AND gold−shuffled **> 0 significant** (paired, ≥3 seeds).
- Secondary: re-run the **role probe** at full scale — expect the +6.8 to hold/grow.

**Steps.**
1. **Materialize the full 10M corpus** → `data/corpora/strict_full/english_base.txt`.
   - Source: HF cache `datasets--BabyLM-community--BabyLM-2026-Strict`. **Verify
     whether it is the 10M Strict-Small or the 100M Strict** (`du -sh`, line/word
     count). If 100M, subsample to ~10M (or accept longer training).
   - Write a tiny prep (`load_dataset(...)` → write one sentence per line) or reuse
     a `prepare_babylm*` if present. Confirm ~10M words.
2. **Build the mixed gold cache with OVERSAMPLED contrast** (the ADR Option-B knob).
   - At 10M BabyLM, the ~16.7k contrast lines (~0.2M tokens) are only ~2% of tokens
     — too dilute for the aux to matter. **Oversample contrast ~6–8×** to reach
     ~12–15% of tokens.
   - Add a `--contrast-repeat N` flag to `scripts/build_contrast_cache.py` (repeat
     the contrast `corpus_with_roles(...)` list N× before `mix_lines`), then:
     `uv run python scripts/build_contrast_cache.py --out-dir data/corpora/contrast_full
     --babylm data/corpora/strict_full/english_base.txt --n-frames 10000 --contrast-repeat 7`
   - **Log the realized contrast token fraction** and the alignment invariant
     (`len(roles) == tokens + lines`).
3. **Train the matrix** (clean config, full corpus): 3 arms × **3 seeds** (5 if
   budget allows). Reuse `run_matrix.sh` with `--base-dir data/corpora/contrast_full`.
   Per-run ≈ 29 min (10M×7ep ÷ 40K tok/s). 9 runs ≈ ~4.5 h; 15 ≈ ~7.5 h. Set TTL ≥ 8 h.
   (Optional recipe-fidelity: grad-accum to effective batch 256 — adds a `--grad-accum`
   flag; not required for the internal comparison.)
4. **Eval each checkpoint:**
   - `uv run python scripts/eval_blimp_pll.py --ckpt <out>/elc.pt --per-paradigm 200 --require-cuda`
   - `uv run python probe_roles.py` (point its checkpoint glob at the full-corpus runs).
5. **Analyze & decide:**
   - **(a) BLiMP targeted gold−baseline ≥ +1.0 AND gold−shuffled significant** →
     **positive downstream translation.** The mechanism pays off. Proceed to write up.
   - **(b) Probe effect holds but BLiMP flat** → **"representational, not downstream"**
     — honest and interesting: the role info is there but pure-MLM doesn't route it
     to the task. Motivates a different downstream head (e.g. fine-tune on a role
     task) or Experiment 2.
   - **(c) Both flat at full scale** → the proxy probe effect doesn't scale;
     reconsider the mechanism.

**Time/cost:** ~5–8 h L4 (~$4–7). **Risks:** corpus-prep correctness (verify counts +
alignment); oversampling ratio is a logged tunable; strict-small BLiMP ceilings are
modest (~65–72 per the paper) so the *gap* matters more than the level.

---

## Experiment 2 — Cross-lingual role transfer (the language-independence test)

**The deep question.** Are kāraka roles encoded in a **language-independent**
subspace — does a role probe trained on one language decode another language's roles?

**Prerequisite that shapes everything:** the current 27 checkpoints are **English-
only** (BabyLM + English contrast); they never saw Sanskrit, so they cannot
represent SA tokens. A *true* cross-lingual probe needs a **bilingual model first.**
So Experiment 2 is staged: a cheap warmup that runs today, then the real build.

### 2-lite — construction-invariance probe (cheap; uses the existing 27 checkpoints)

A within-English proxy for language-independence: does the role subspace survive the
**active↔passive** surface change?
- Build the probe (clone `probe_roles.py`) but **train on ACTIVE sentences only,
  test on PASSIVE** (same roles, flipped surface). Source clean labels from our gold
  contrast active/passive realizations (`build_contrast_corpus` gives both), or the
  COGS `active_to_passive` / `passive_to_active` categories.
- Compare gold vs baseline vs shuffled. **If gold's role subspace is voice-invariant,
  train-active→test-passive accuracy stays high for gold but drops for baseline.**
- Cost: ~15 min, existing checkpoints. **Decision:** gold transfers across voice >
  baseline → role subspace is construction-invariant (a necessary condition for
  cross-lingual) → green-light 2-full.

### 2-full — bilingual model + cross-lingual probe (the headline result)

1. **Tokenizer.** The 8000-vocab English SP doesn't cover Sanskrit SLP1. Train a new
   SentencePiece tokenizer on **EN + SA** text (or extend the vocab) so SA tokens are
   representable. (New artifact; `SentencePieceTrainer` is in the repo.)
2. **Bilingual gold corpus.** Generate SA contrast realizations (the Vidyut
   `VidyutFrameRealizer`, gold `karaka_parse`) + the EN contrast + BabyLM. **Extend
   `contrast_corpus.py`** to emit SA lines + SA per-token role ids (the SA realizer's
   `karaka_parse` → per-token roles via the new tokenizer; reuse `align_pieces_to_role_ids`).
   The aux then trains role prediction on **both** languages.
3. **Train** the bilingual baseline/gold/shuffled matrix (≥3 seeds).
4. **Cross-lingual probe.** Train the role probe on **English** reps, test on
   **Sanskrit** reps (and vice versa). **If gold's English-trained probe decodes
   Sanskrit roles above baseline/shuffled, roles are encoded language-independently
   in the gold model** — the direct evidence for the Pāṇinian claim, and the paper's
   centerpiece.

**Time/cost:** 2-lite ~15 min (today). 2-full is a bigger build — bilingual tokenizer
(~30 min) + bilingual cache (new code, ~half a day dev) + training (~5 h) + probe.
Plan 1–2 sessions. **Decision:** EN→SA probe transfer significantly above
baseline/shuffled → language-independent role encoding confirmed.

---

## Recommended order

1. **Experiment 1** (full-corpus) — well-scoped, one session, tests the most
   immediate open question, reuses everything baked in the AMI.
2. **Experiment 2-lite** (construction-invariance) — cheap; run it in the *same*
   session as Exp 1 (existing checkpoints) to de-risk the cross-lingual idea.
3. **Experiment 2-full** (bilingual cross-lingual) — the deepest result and the
   biggest build; pursue it if 1 + 2-lite are encouraging.

Honest framing throughout: report whichever of positive / representational-only /
null actually lands, with the Tarka memo (strongest objection to your own result),
per the closure contract.
