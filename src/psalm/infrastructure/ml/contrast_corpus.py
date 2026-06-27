"""Gold kāraka role-contrast corpus + aux role-label stream (H1_CONTRAST, ADR-0043).

Realizes role-contrast sets into English sentences and the matching per-token
**gold** role-label stream the śābdabodha auxiliary head consumes (Option B —
mixed corpus). Reuses :func:`align_pieces_to_role_ids` and ``SHABDABODHA_LABELS``,
so the ``.bin`` contract is identical to ``build_shabdabodha_cache`` — the only
difference is that the labels are gold-by-construction (from the frame realizer)
rather than recovered from a spaCy parse, removing the label noise.

The realizer emits roles in WX (``karwA``, ``karaNam``, …); :data:`WX_TO_SHABDABODHA`
translates them to the head's romanized label set.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from psalm.domain.data.karaka_frames import KarakaFrame
from psalm.infrastructure.generators.english_frame_realizer import EnglishFrameRealizer
from psalm.infrastructure.ml.shabdabodha_target import (
    SHABDABODHA_LABELS,
    align_pieces_to_role_ids,
    role_to_id,
)

#: WX role name (realizer / construction-index vocabulary) → śābdabodha label name.
WX_TO_SHABDABODHA: dict[str, str] = {
    "karwA": "karta",
    "karma": "karma",
    "karaNam": "karana",
    "sampraxAnam": "sampradana",
    "apAxAnam": "apadana",
    "aXikaraNam": "adhikarana",
    "kriyA": "kriya",
    "separator": "separator",
}


def wx_role_id(wx_role: str) -> int:
    """Map a WX role name (or ``separator``) to a śābdabodha label id; unknown → ``none``."""
    return role_to_id(WX_TO_SHABDABODHA.get(wx_role, "none"))


def sentence_role_ids(pieces: list[str], word_roles: list[tuple[str, str]]) -> list[int]:
    """Per-piece role ids for one sentence, from its gold per-word roles.

    Aligns word-order roles to SentencePiece pieces via the shared
    :func:`align_pieces_to_role_ids` (first piece of each word carries the role;
    continuation pieces and punctuation are ``separator``). Returns one id per
    piece.
    """
    role_names = [WX_TO_SHABDABODHA.get(role, "none") for _, role in word_roles]
    return align_pieces_to_role_ids(pieces, role_names)


#: One corpus line paired with its per-piece role ids (no EOS role).
LineRoles = tuple[str, list[int]]


def corpus_with_roles(
    frames: Iterable[KarakaFrame],
    encode_pieces: Callable[[str], list[str]],
    *,
    realizer: EnglishFrameRealizer | None = None,
) -> list[LineRoles]:
    """Per-line ``(text, per-piece gold role ids)`` for the contrast corpus.

    Each frame contributes its active and passive members; lines that cannot be
    realized are skipped. No EOS role is appended here (see :func:`flatten_corpus`).
    """
    realize = realizer or EnglishFrameRealizer()
    out: list[LineRoles] = []
    for frame in frames:
        for voice in ("active", "passive"):
            sentence = realize.realize(frame, voice=voice)
            word_roles = realize.word_roles(frame, voice=voice)
            if sentence is None or word_roles is None:
                continue
            ids = sentence_role_ids(encode_pieces(sentence.text), word_roles)
            out.append((sentence.text, ids))
    return out


def none_role_lines(
    lines: Iterable[str], encode_pieces: Callable[[str], list[str]]
) -> list[LineRoles]:
    """Background (e.g. BabyLM) lines labeled entirely ``none`` (ADR-0043 Option B).

    The model still learns MLM on these tokens; the aux head sees ``none`` (no
    salient role) there, confining role supervision to the contrast slice.
    """
    none_id = SHABDABODHA_LABELS["none"]
    return [(line, [none_id] * len(encode_pieces(line))) for line in lines if line]


def mix_lines(
    contrast: list[LineRoles], background: list[LineRoles], *, seed: int
) -> list[LineRoles]:
    """Deterministically interleave contrast + background lines (seeded shuffle)."""
    combined = [*contrast, *background]
    order = np.random.default_rng(seed).permutation(len(combined))
    return [combined[int(i)] for i in order]


def flatten_corpus(
    pairs: list[LineRoles], *, with_eos_role: bool = True
) -> tuple[list[str], list[int]]:
    """Flatten per-line pairs into ``(lines, role stream)`` with optional EOS role.

    With ``with_eos_role`` a ``separator`` is appended after each line, matching
    ``TokenPacker``'s per-line EOS so ``len(roles) == total_tokens + n_lines``.
    """
    sep = SHABDABODHA_LABELS["separator"]
    lines: list[str] = []
    roles: list[int] = []
    for text, ids in pairs:
        lines.append(text)
        roles.extend(ids)
        if with_eos_role:
            roles.append(sep)
    return lines, roles


def build_contrast_corpus(
    frames: Iterable[KarakaFrame],
    encode_pieces: Callable[[str], list[str]],
    *,
    with_eos_role: bool = True,
    realizer: EnglishFrameRealizer | None = None,
) -> tuple[list[str], list[int]]:
    """Realize a contrast-only corpus and its aligned role stream (no background)."""
    pairs = corpus_with_roles(frames, encode_pieces, realizer=realizer)
    return flatten_corpus(pairs, with_eos_role=with_eos_role)


def shuffle_roles(roles: list[int], *, seed: int) -> list[int]:
    """Permute the role-label stream — the ADR-0043 shuffled-role specificity control.

    Preserves the label multiset (same role distribution) but destroys token
    alignment, so any aux benefit that survives shuffling is *not* alignment-
    specific. Deterministic given ``seed``.
    """
    arr = np.asarray(roles, dtype=np.int64)
    np.random.default_rng(seed).shuffle(arr)
    return [int(x) for x in arr]
