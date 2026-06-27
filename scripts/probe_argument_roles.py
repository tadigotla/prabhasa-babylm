"""Argument-role linear probe (Step 1, Option A) for the H1_CONTRAST checkpoints.

Tests whether gold role-contrast supervision made argument roles (agent/theme/
recipient) more *linearly decodable* from the frozen encoder — the mechanism's
actual claim — and whether that generalizes to natural held-out COGS sentences.

For each checkpoint: extract per-token hidden states on COGS sentences, label the
argument tokens from the COGS logical form (x_N = token index N), train a linear
probe on COGS-train reps, test on COGS-GEN reps. Encoder frozen; only the probe
trains. Compare probe accuracy across baseline / gold / shuffled.
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
N_TRAIN, N_TEST = 1400, 900
WORD_START = "▁"  # SentencePiece ▁

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


def train_probe(Xtr, ytr, Xte, yte):
    Xtr = torch.tensor(Xtr, device=DEVICE)
    Xte = torch.tensor(Xte, device=DEVICE)
    mu, sdv = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sdv, (Xte - mu) / sdv
    ytr = torch.tensor(ytr, device=DEVICE)
    yte = torch.tensor(yte, device=DEVICE)
    torch.manual_seed(0)
    probe = nn.Linear(768, 3).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(400):
        opt.zero_grad()
        nn.functional.cross_entropy(probe(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        pred = probe(Xte).argmax(1)
        acc3 = (pred == yte).float().mean().item()
        m = yte < 2  # agent vs theme only
        acc_at = (pred[m] == yte[m]).float().mean().item()
        maj = max((yte == c).float().mean().item() for c in range(3))  # majority baseline
    return acc3, acc_at, maj, len(ytr), len(yte)


train_rows = load_cogs("train", allow_download=True)[: N_TRAIN * 2]
gen_rows = load_cogs("gen", allow_download=True)[: N_TEST * 3]

results = {}
for arm in ["baseline", "gold", "shuffled"]:
    for seed in [0, 1, 2]:
        model, _ = load_elc_checkpoint(Path(f"runs/contrast_matrix/{arm}_s{seed}/elc.pt"), device=DEVICE)
        model.eval()
        Xtr, ytr = collect(train_rows[:N_TRAIN], model)
        Xte, yte = collect(gen_rows[:N_TEST], model)
        acc3, acc_at, maj, ntr, nte = train_probe(Xtr, ytr, Xte, yte)
        results[f"{arm}_s{seed}"] = dict(acc3=acc3, acc_agent_theme=acc_at, majority=maj, ntr=ntr, nte=nte)
        print(f"{arm}_s{seed}: 3class={acc3:.4f} agent-theme={acc_at:.4f} maj={maj:.3f} (tr={ntr} te={nte})", flush=True)

json.dump(results, open("runs/contrast_matrix/probe_results.json", "w"), indent=2)
for metric in ["acc3", "acc_agent_theme"]:
    print(f"\n=== {metric} (mean over seeds) ===")
    for arm in ["baseline", "gold", "shuffled"]:
        vals = [results[f"{arm}_s{s}"][metric] for s in [0, 1, 2]]
        print(f"  {arm:9} {st.mean(vals) * 100:.2f}   seeds={[round(v * 100, 2) for v in vals]}")
