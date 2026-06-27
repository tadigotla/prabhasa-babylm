"""Aux-head wiring contract for the gold contrast role stream (task 5, slice 3).

Drives a gold role-label stream through the *exact* training seam the F3 path
uses — ``RoleStreamPacker`` → ``ShabdabodhaHead`` → ``shabdabodha_aux_loss`` —
and asserts the shapes and a finite loss. This proves our gold ``.bin`` is a
drop-in for the existing, generic aux machinery (no training-loop changes).

Gated on torch; skips on hosts without it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from psalm.domain.data.karaka_frames import enumerate_frames  # noqa: E402
from psalm.infrastructure.ml.contrast_corpus import build_contrast_corpus  # noqa: E402
from psalm.infrastructure.ml.packing import RoleStreamPacker  # noqa: E402
from psalm.infrastructure.ml.shabdabodha_head import (  # noqa: E402
    ShabdabodhaHead,
    shabdabodha_aux_loss,
)
from psalm.infrastructure.ml.shabdabodha_target import N_LABELS  # noqa: E402


def _fake_pieces(text: str) -> list[str]:
    return ["▁" + w.lower().rstrip(".") for w in text.split()]


def test_gold_roles_drive_a_finite_aux_loss() -> None:
    # 1. Build the gold role stream and store it as the uint8 the .bin uses.
    _, roles = build_contrast_corpus(list(enumerate_frames(60, seed=0)), _fake_pieces)
    role_arr = np.array(roles, dtype=np.uint8)
    assert role_arr.max() < N_LABELS

    # 2. Window it exactly as the trainer does.
    seq_len, batch_size, d_model = 16, 4, 32
    packer = RoleStreamPacker(role_arr, seq_len=seq_len)
    role_batch = next(packer.packed_batches(1, batch_size, device="cpu"))
    assert role_batch.shape == (batch_size, seq_len)
    assert role_batch.dtype == torch.long
    assert int(role_batch.min()) >= 0 and int(role_batch.max()) < N_LABELS

    # 3. Run the aux head + loss on (stand-in) encoder hidden states.
    head = ShabdabodhaHead(d_model)
    hidden = torch.randn(batch_size, seq_len, d_model)
    logits = head(hidden)
    assert logits.shape == (batch_size, seq_len, N_LABELS)

    loss = shabdabodha_aux_loss(logits, role_batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_aux_loss_responds_to_role_labels() -> None:
    # A sanity check that the labels actually drive the loss: perfect logits for
    # the gold labels give a far lower loss than uniform logits.
    _, roles = build_contrast_corpus(list(enumerate_frames(40, seed=1)), _fake_pieces)
    seq_len, batch_size = 16, 4
    role_batch = next(
        RoleStreamPacker(np.array(roles, dtype=np.uint8), seq_len=seq_len).packed_batches(
            1, batch_size, device="cpu"
        )
    )
    uniform = torch.zeros(batch_size, seq_len, N_LABELS)
    confident = torch.full((batch_size, seq_len, N_LABELS), -10.0)
    confident.scatter_(2, role_batch.unsqueeze(-1), 10.0)  # peak at the gold label
    loss_uniform = shabdabodha_aux_loss(uniform, role_batch)
    loss_confident = shabdabodha_aux_loss(confident, role_batch)
    assert loss_confident < loss_uniform
