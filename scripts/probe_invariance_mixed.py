"""Mixed-voice diagnostic for Step 0 — resolves the active-only-training confound.

The active->passive probe (probe_invariance.py) collapses below chance for ALL arms:
trained on actives alone, the linear role axis is position-tied (subject=agent), so
it inverts on passives. That cannot distinguish "no invariant role content" from
"invariant content the active-only probe can't isolate" (position perfectly predicts
role in the active-only training set, so a linear probe never needs the invariant
axis even if it exists).

This script DECORRELATES position from role in training: the probe trains on a
BALANCED mix of actives + passives (in passives, subject=theme, so position no longer
predicts role). If a position-robust role axis exists in the encoder, the probe can
now find it and should clear chance on held-out passives. The decisive cross-arm
question: on held-out PASSIVES, does gold beat baseline? If yes -> gold encodes
invariant role content the active-only test couldn't see (reconsider STOP). If
gold ~= baseline -> gold's advantage is active-positional only (STOP confirmed).

  probe-train  : 520 simple ACTIVES (COGS train) + 520 simple PASSIVES (COGS gen)  [balanced]
  test-active  : held-out simple ACTIVES (COGS gen)
  test-passive : held-out simple PASSIVES (COGS gen, disjoint from the trained 520)

Primary readout: agent-vs-theme accuracy on test-passive (chance 50%).
Extraction identical to probe_roles.py / probe_invariance.py.
"""

import json
import re
import statistics as st
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
WORD_START = "▁"
SEEDS = list(range(9))
ARMS = ["baseline", "gold", "shuffled"]
N_BAL = 520  # balanced count per voice in the training mix

sp = spm.SentencePieceProcessor()
sp.Load(TOK)
ROLE_RE = re.compile(r"[A-Za-z]+\.(agent|theme|recipient)\((?:x_\d+|[A-Za-z]+),(x_\d+|[A-Za-z]+)\)")


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
    if word_idx + 1 != len(sent.split()):
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


def is_simple_passive(sent):
    t = [w.lower() for w in sent.split()]
    return ("was" in t or "were" in t) and "by" in t and "that" not in t and "to" not in t


def is_simple_active(sent):
    t = [w.lower() for w in sent.split()]
    return "was" not in t and "were" not in t and "by" not in t and "that" not in t and "to" not in t


def build_pool(rows, voice_pred, cap):
    pool = []
    for sent, lf in rows:
        if not voice_pred(sent):
            continue
        r = sent_examples(sent, lf)
        if r is None:
            continue
        present = {role for _, role in r[1]}
        if 0 in present and 1 in present:
            pool.append((sent, lf))
            if len(pool) >= cap:
                break
    return pool


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
        m = yte < 2
        acc_at = (pred[m] == yte[m]).float().mean().item() if int(m.sum()) else float("nan")
    return acc_at, int(m.sum())


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0.0
    se = sd / n**0.5 if n else 0.0
    t = m / se if se > 0 else float("inf")
    ci = (m - 2.306 * se, m + 2.306 * se)
    return dict(mean=m, t=t, ci=ci, pos=sum(1 for x in d if x > 0), n=n)


# ---- pools -----------------------------------------------------------------------
train_rows = load_cogs("train", allow_download=True)
gen_rows = load_cogs("gen", allow_download=True)

actives = build_pool(train_rows, is_simple_active, N_BAL)        # 520 actives (COGS train)
passives = build_pool(gen_rows, is_simple_passive, 1000)         # all gen passives
test_active = build_pool(gen_rows, is_simple_active, 1000)       # held-out actives (COGS gen)

p_train = passives[:N_BAL]          # 520 passives into the training mix
p_test = passives[N_BAL:]           # remainder held out (disjoint sentences)
mix_train = actives + p_train       # balanced: 520 active + 520 passive

print(f"pools: mix_train={len(mix_train)} (act {len(actives)}+pas {len(p_train)})  "
      f"test_active={len(test_active)}  test_passive={len(p_test)}", flush=True)

results = {}
for arm in ARMS:
    for seed in SEEDS:
        model, _ = load_elc_checkpoint(Path(f"runs/contrast_matrix/{arm}_s{seed}/elc.pt"), device=DEVICE)
        model.eval()
        Xtr, ytr = collect(mix_train, model)
        XaA, yaA = collect(test_active, model)
        XpP, ypP = collect(p_test, model)
        probe, mu, sdv = fit_probe(Xtr, ytr)
        at_act, n_act = eval_probe(probe, mu, sdv, XaA, yaA)
        at_pas, n_pas = eval_probe(probe, mu, sdv, XpP, ypP)
        results[f"{arm}_s{seed}"] = dict(at_active=at_act, at_passive=at_pas, n_pas=n_pas)
        print(f"{arm}_s{seed}: mixed-train -> heldout ACTIVE at={at_act*100:.1f} | heldout PASSIVE at={at_pas*100:.1f} (n_pas={n_pas})", flush=True)

json.dump(results, open("runs/contrast_matrix/probe_invariance_mixed_results.json", "w"), indent=2)

print("\n" + "=" * 78)
print("MIXED-VOICE DIAGNOSTIC (train balanced active+passive), agent-vs-theme acc")
print("=" * 78)
for key, label in [("at_passive", "held-out PASSIVE (decisive: position decorrelated)"),
                   ("at_active", "held-out ACTIVE")]:
    print(f"\n--- {label} ---")
    means = {}
    for arm in ARMS:
        vals = [results[f"{arm}_s{s}"][key] * 100 for s in SEEDS]
        means[arm] = vals
        print(f"  {arm:9} {st.mean(vals):.2f}   n={len(vals)}  seeds={[round(v,1) for v in vals]}")
    for hi, lo in [("gold", "baseline"), ("gold", "shuffled"), ("shuffled", "baseline")]:
        r = paired(means[hi], means[lo])
        print(f"  {hi}-{lo:9} {r['mean']:+.2f}  CI[{r['ci'][0]:+.1f},{r['ci'][1]:+.1f}]  t={r['t']:.2f}  {r['pos']}/{r['n']} seeds +")
