"""Construction-invariance probe (Step 0) for the H1_CONTRAST checkpoints.

Tests the crux abstraction claim behind the n=9 role-probe result: is the gold
argument-role subspace **invariant to a surface voice change**? Train a linear role
probe on ACTIVE sentences, test it on PASSIVE sentences (same agent/theme roles,
surface positions flipped — agent moves from subject to a `by`-phrase, theme from
object to subject). A probe that learned a genuine role subspace transfers; one that
learned "subject position = agent" fails (scores at/below chance on passives, where
the subject is the theme).

Data: COGS only — held out from every arm (no arm ever trained on COGS), so the test
is fair across baseline / gold / shuffled. The gold/shuffled arms *did* train on the
contrast active/passive realizations via the aux head, so testing invariance on those
would be circular; COGS avoids that.

  probe-train  : simple transitive ACTIVES from COGS train      (NP VERBed NP)
  test-active  : simple transitive ACTIVES from COGS gen (held) -> within-voice ceiling
  test-passive : simple long PASSIVES  from COGS gen (held)     (NP was VERBed by NP)

Primary metric: agent-vs-theme accuracy (the kartā/karma crux; balanced -> 50% floor).
3-class reported too. Compare active->passive across baseline / gold / shuffled over
9 seeds, paired (gold-baseline, gold-shuffled, shuffled-baseline).

Extraction (sent_examples / collect / linear probe / normalisation / 400 steps /
probe seed 0) is byte-identical to probe_roles.py so the numbers are comparable.

Usage:  python probe_invariance.py --dry-run   # build pools, print sizes+samples, no GPU
        python probe_invariance.py             # full run over 27 checkpoints
"""

import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn

from psalm.infrastructure.eval.cogs import load_cogs
from psalm.infrastructure.ml.elc_trainer import load_elc_checkpoint

DEVICE = "cuda"
TOK = "data/tokenizer/strict_small/spm.model"
ROLES = {"agent": 0, "theme": 1, "recipient": 2}
WORD_START = "▁"  # SentencePiece ▁
SEEDS = list(range(9))
ARMS = ["baseline", "gold", "shuffled"]

# Pool caps (sentences). Each usable sentence yields ~2 role tokens (agent+theme).
CAP_TRAIN_ACTIVE = 1600
CAP_TEST_ACTIVE = 1000
CAP_TEST_PASSIVE = 1000

sp = spm.SentencePieceProcessor()
sp.Load(TOK)
ROLE_RE = re.compile(r"[A-Za-z]+\.(agent|theme|recipient)\((?:x_\d+|[A-Za-z]+),(x_\d+|[A-Za-z]+)\)")


# ---- role extraction (verbatim from probe_roles.py) -------------------------------
def parse_roles(sent, lf):
    toks = sent.split()
    out = {}
    for m in ROLE_RE.finditer(lf.replace(" ", "")):
        role, filler = m.group(1), m.group(2)
        idx = int(filler[2:]) if filler.startswith("x_") else (toks.index(filler) if filler in toks else -1)
        if 0 <= idx < len(toks):
            out[idx] = ROLES[role]
    return out


def sent_examples(sent, lf):
    roles = parse_roles(sent, lf)
    if not roles:
        return None
    pieces = sp.EncodeAsPieces(sent)
    ids = sp.EncodeAsIds(sent)
    word_idx, hits = -1, []
    for i, p in enumerate(pieces):
        if i == 0 or p.startswith(WORD_START):
            word_idx += 1
            if word_idx in roles:
                hits.append((i, roles[word_idx]))
    if word_idx + 1 != len(sent.split()):  # SP/COGS tokenisation diverged → skip
        return None
    return ids, hits


def collect(rows, model):
    X, y = [], []
    for sent, lf in rows:
        r = sent_examples(sent, lf)
        if r is None:
            continue
        ids, hits = r
        with torch.no_grad():
            hid = model(torch.tensor([ids], device=DEVICE))[1]["hidden_mlm"][0]
        for pos, role in hits:
            X.append(hid[pos].float().cpu().numpy())
            y.append(role)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ---- voice classification + pool building ----------------------------------------
def is_simple_passive(sent):
    """Long passive, single clause: 'NP was/were VERBed by NP .' (excludes embedded)."""
    t = [w.lower() for w in sent.split()]
    return (("was" in t or "were" in t) and "by" in t and "that" not in t and "to" not in t)


def is_simple_active(sent):
    """Simple transitive active: no passive aux, no by-phrase, single clause."""
    t = [w.lower() for w in sent.split()]
    return ("was" not in t and "were" not in t and "by" not in t and "that" not in t and "to" not in t)


def build_pool(rows, voice_pred, cap):
    """Keep sentences passing voice_pred whose roles include BOTH agent and theme."""
    pool = []
    for sent, lf in rows:
        if not voice_pred(sent):
            continue
        r = sent_examples(sent, lf)
        if r is None:
            continue
        present = {role for _, role in r[1]}
        if 0 in present and 1 in present:  # agent (0) AND theme (1)
            pool.append((sent, lf))
            if len(pool) >= cap:
                break
    return pool


# ---- linear probe (fit once on actives, eval on multiple test sets) --------------
def fit_probe(Xtr, ytr):
    Xtr = torch.tensor(Xtr, device=DEVICE)
    mu, sdv = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mu) / sdv
    ytr = torch.tensor(ytr, device=DEVICE)
    torch.manual_seed(0)
    probe = nn.Linear(768, 3).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(400):
        opt.zero_grad()
        nn.functional.cross_entropy(probe(Xtr), ytr).backward()
        opt.step()
    return probe, mu, sdv


def eval_probe(probe, mu, sdv, Xte, yte):
    Xte = (torch.tensor(Xte, device=DEVICE) - mu) / sdv
    yte = torch.tensor(yte, device=DEVICE)
    with torch.no_grad():
        pred = probe(Xte).argmax(1)
        acc3 = (pred == yte).float().mean().item()
        m = yte < 2  # agent vs theme only
        acc_at = (pred[m] == yte[m]).float().mean().item() if int(m.sum()) else float("nan")
        maj_at = max((yte[m] == c).float().mean().item() for c in range(2)) if int(m.sum()) else float("nan")
    return dict(acc3=acc3, acc_at=acc_at, maj_at=maj_at, n_at=int(m.sum()), n=len(yte))


def paired(a, b):
    """Paired difference a-b across seeds: mean, t, 95% CI (df=8), sign count."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0.0
    se = sd / n**0.5 if n else 0.0
    t = m / se if se > 0 else float("inf")
    tcrit = 2.306  # t_0.975, df=8
    ci = (m - tcrit * se, m + tcrit * se)
    pos = sum(1 for x in d if x > 0)
    return dict(mean=m, t=t, ci=ci, pos=pos, n=n)


# ---- build pools -----------------------------------------------------------------
train_rows = load_cogs("train", allow_download=True)
gen_rows = load_cogs("gen", allow_download=True)

train_active = build_pool(train_rows, is_simple_active, CAP_TRAIN_ACTIVE)
test_active = build_pool(gen_rows, is_simple_active, CAP_TEST_ACTIVE)
test_passive = build_pool(gen_rows, is_simple_passive, CAP_TEST_PASSIVE)

print(f"pools: train_active={len(train_active)}  test_active={len(test_active)}  test_passive={len(test_passive)}")

if "--dry-run" in sys.argv:
    for name, pool in [("train_active", train_active), ("test_active", test_active), ("test_passive", test_passive)]:
        print(f"\n=== {name} samples ===")
        for sent, lf in pool[:5]:
            print(f"  {sent}")
            print(f"     roles={parse_roles(sent, lf)}")
    sys.exit(0)

# ---- run over 27 checkpoints -----------------------------------------------------
results = {}
for arm in ARMS:
    for seed in SEEDS:
        model, _ = load_elc_checkpoint(Path(f"runs/contrast_matrix/{arm}_s{seed}/elc.pt"), device=DEVICE)
        model.eval()
        Xtr, ytr = collect(train_active, model)
        XaA, yaA = collect(test_active, model)
        XpP, ypP = collect(test_passive, model)
        probe, mu, sdv = fit_probe(Xtr, ytr)
        ev_act = eval_probe(probe, mu, sdv, XaA, yaA)  # active->active ceiling
        ev_pas = eval_probe(probe, mu, sdv, XpP, ypP)  # active->passive crux
        results[f"{arm}_s{seed}"] = dict(active=ev_act, passive=ev_pas, ntr=len(ytr))
        print(
            f"{arm}_s{seed}: A->A at={ev_act['acc_at']*100:.1f} 3c={ev_act['acc3']*100:.1f} | "
            f"A->P at={ev_pas['acc_at']*100:.1f} 3c={ev_pas['acc3']*100:.1f} "
            f"(maj_at={ev_pas['maj_at']*100:.1f}, n_at={ev_pas['n_at']}, ntr={len(ytr)})",
            flush=True,
        )

json.dump(results, open("runs/contrast_matrix/probe_invariance_results.json", "w"), indent=2)

# ---- aggregate -------------------------------------------------------------------
def arm_vals(arm, split, key):
    return [results[f"{arm}_s{s}"][split][key] for s in SEEDS]

print("\n" + "=" * 78)
print("CONSTRUCTION-INVARIANCE (train ACTIVE -> test PASSIVE), agent-vs-theme acc")
print("=" * 78)
for split, label in [("passive", "active->PASSIVE (crux)"), ("active", "active->active (ceiling)")]:
    print(f"\n--- {label} : agent-vs-theme ---")
    means = {}
    for arm in ARMS:
        vals = [v * 100 for v in arm_vals(arm, split, "acc_at")]
        means[arm] = vals
        print(f"  {arm:9} {st.mean(vals):.2f}   n={len(vals)}  seeds={[round(v,1) for v in vals]}")
    for hi, lo in [("gold", "baseline"), ("gold", "shuffled"), ("shuffled", "baseline")]:
        r = paired(means[hi], means[lo])
        print(
            f"  {hi}-{lo:9} {r['mean']:+.2f}  CI[{r['ci'][0]:+.1f},{r['ci'][1]:+.1f}]  "
            f"t={r['t']:.2f}  {r['pos']}/{r['n']} seeds +"
        )

# invariance gap: how much each arm DROPS going active->passive (lower drop = more invariant)
print("\n--- invariance gap = (active->active) - (active->passive), agent-vs-theme (lower = more invariant) ---")
gaps = {}
for arm in ARMS:
    g = [(arm_vals(arm, "active", "acc_at")[i] - arm_vals(arm, "passive", "acc_at")[i]) * 100 for i in range(len(SEEDS))]
    gaps[arm] = g
    print(f"  {arm:9} {st.mean(g):+.2f}   seeds={[round(v,1) for v in g]}")
for hi, lo in [("gold", "baseline"), ("gold", "shuffled")]:
    r = paired(gaps[lo], gaps[hi])  # baseline_gap - gold_gap > 0 means gold is MORE invariant
    print(
        f"  ({lo}_gap - {hi}_gap) {r['mean']:+.2f}  CI[{r['ci'][0]:+.1f},{r['ci'][1]:+.1f}]  "
        f"t={r['t']:.2f}  {r['pos']}/{r['n']} seeds +  (positive => {hi} more invariant)"
    )
