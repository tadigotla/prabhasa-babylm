"""Śābdabodha verbal-cognition targets (RQ-B / SPEC 0003).

Token-level kāraka-relation labels — the śābdabodha role of each token with
respect to the clause's kriyā (verbal action / mukhya-viśeṣya). Built from REAL
spaCy dependency parses (reusing ``english_karaka_real.parse_and_assign``) and
aligned to SentencePiece pieces: the first piece of each word carries the word's
role; continuation pieces are ``separator``.

This is the supervised target for the RQ-B auxiliary objective (predict the
viśiṣṭa-jñāna relational structure), multi-tasked with pure-MLM. No synthetic
data — every label derives from a real parse.

Alignment note (v1): word→piece alignment uses the SentencePiece ▁ word-start
marker, walked in parallel with spaCy tokens in order. This is an approximation
where spaCy and SentencePiece tokenisation diverge (contractions, punctuation);
SPEC 0003's audit step quantifies the residual label noise. A char-offset
alignment is the planned refinement if noise > 5%.
"""

from __future__ import annotations

# 10-class label set (SPEC 0003). 'unknown' (from the parser) maps to 'none'.
SHABDABODHA_LABELS: dict[str, int] = {
    "karta": 0,
    "karma": 1,
    "karana": 2,
    "sampradana": 3,
    "apadana": 4,
    "adhikarana": 5,
    "visesana": 6,
    "kriya": 7,
    "separator": 8,
    "none": 9,
}
N_LABELS = len(SHABDABODHA_LABELS)
IGNORE_INDEX = -100  # for positions with no target (e.g., pad)
_WORD_START = "▁"  # ▁ SentencePiece word-boundary prefix


def role_to_id(role: str) -> int:
    """Map a kāraka role name → label id; 'unknown'/missing → 'none'."""
    return SHABDABODHA_LABELS.get(role, SHABDABODHA_LABELS["none"])


def align_pieces_to_role_ids(pieces: list[str], role_names: list[str]) -> list[int]:
    """Align word-order role names to SentencePiece pieces.

    First piece of each word (▁-prefixed, or piece 0) takes the next word's role;
    continuation pieces are 'separator'. Returns one id per piece. Shared by the
    single-sentence builder and the offline corpus cache generator (no duplication).
    """
    roles_iter = iter(role_names)
    sep = SHABDABODHA_LABELS["separator"]
    out: list[int] = []
    for i, piece in enumerate(pieces):
        if i == 0 or piece.startswith(_WORD_START):
            try:
                out.append(role_to_id(next(roles_iter)))
            except StopIteration:
                out.append(SHABDABODHA_LABELS["none"])
        else:
            out.append(sep)
    return out


class ShabdabodhaTargetBuilder:
    """Builds per-SentencePiece-token śābdabodha role labels from real parses."""

    def __init__(self, nlp: object, sp: object) -> None:
        self.nlp = nlp
        self.sp = sp

    def build_labels(self, sentence: str) -> list[int]:
        """Return one role-id per SentencePiece piece of ``sentence``.

        First piece of each word = the word's kāraka role; continuation pieces =
        ``separator``. Length matches ``sp.EncodeAsPieces(sentence)``.
        """
        # Lazy import: spaCy is only needed for the parser-based builder, not for
        # the label constants / alignment helper this module also exports.
        from psalm.domain.linguistics.english_karaka_real import parse_and_assign

        word_roles = parse_and_assign(sentence, self.nlp)  # [TokenRole(text, role)]
        pieces = self.sp.EncodeAsPieces(sentence)  # type: ignore[attr-defined]
        return align_pieces_to_role_ids(pieces, [tr.role for tr in word_roles])
