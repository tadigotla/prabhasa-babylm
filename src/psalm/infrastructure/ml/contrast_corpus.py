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


def build_contrast_corpus(
    frames: Iterable[KarakaFrame],
    encode_pieces: Callable[[str], list[str]],
    *,
    with_eos_role: bool = True,
    realizer: EnglishFrameRealizer | None = None,
) -> tuple[list[str], list[int]]:
    """Realize a gold contrast corpus and its aligned role-label stream.

    For every frame, realize the active and passive members of its contrast set;
    each yields a text line and its per-piece gold role ids. With
    ``with_eos_role`` a ``separator`` is appended after each line, matching
    ``TokenPacker``'s per-line EOS so the role stream stays positionally 1:1 with
    the trainer's token stream (``len(roles) == total_tokens + n_lines``).

    Returns ``(lines, role_ids)``; ``role_ids`` is one flat list over all lines.
    """
    realize = realizer or EnglishFrameRealizer()
    lines: list[str] = []
    roles: list[int] = []
    sep = SHABDABODHA_LABELS["separator"]
    for frame in frames:
        for voice in ("active", "passive"):
            sentence = realize.realize(frame, voice=voice)
            word_roles = realize.word_roles(frame, voice=voice)
            if sentence is None or word_roles is None:
                continue
            pieces = encode_pieces(sentence.text)
            roles.extend(sentence_role_ids(pieces, word_roles))
            if with_eos_role:
                roles.append(sep)
            lines.append(sentence.text)
    return lines, roles


def shuffle_roles(roles: list[int], *, seed: int) -> list[int]:
    """Permute the role-label stream — the ADR-0043 shuffled-role specificity control.

    Preserves the label multiset (same role distribution) but destroys token
    alignment, so any aux benefit that survives shuffling is *not* alignment-
    specific. Deterministic given ``seed``.
    """
    arr = np.asarray(roles, dtype=np.int64)
    np.random.default_rng(seed).shuffle(arr)
    return [int(x) for x in arr]
