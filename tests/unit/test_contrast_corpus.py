"""Unit tests for the gold contrast corpus + aux role-label stream (task 5).

Builds the per-token gold role-label stream the śābdabodha aux head consumes
(Option B), reusing the same alignment primitive and label set as
``build_shabdabodha_cache`` — but the labels are gold-by-construction, not parsed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from psalm.domain.data.karaka_frames import KarakaFrame, enumerate_frames
from psalm.infrastructure.generators.english_frame_realizer import EnglishFrameRealizer
from psalm.infrastructure.ml.contrast_corpus import (
    WX_TO_SHABDABODHA,
    build_contrast_corpus,
    sentence_role_ids,
    shuffle_roles,
    wx_role_id,
)
from psalm.infrastructure.ml.shabdabodha_target import SHABDABODHA_LABELS

_SPM = Path("data/tokenizer/strict_small/spm.model")


def _instrument_frame() -> KarakaFrame:
    return KarakaFrame(
        {
            "words": [
                {
                    "id": 1,
                    "pos": "noun",
                    "stem": "nara",
                    "gender": "puM",
                    "number": "eka",
                    "karaka": "karwA",
                    "head": 4,
                },
                {
                    "id": 2,
                    "pos": "noun",
                    "stem": "Pala",
                    "gender": "napuM",
                    "number": "eka",
                    "karaka": "karma",
                    "head": 4,
                },
                {
                    "id": 3,
                    "pos": "noun",
                    "stem": "puswaka",
                    "gender": "napuM",
                    "number": "eka",
                    "karaka": "karaNam",
                    "head": 4,
                },
                {
                    "id": 4,
                    "pos": "verb",
                    "dhatu": "KAx1",
                    "prayoga": "karwari",
                    "lakara": "varwamAnaH",
                },
            ]
        },
        ("instr",),
    )


def _fake_pieces(text: str) -> list[str]:
    """Toy SentencePiece: each whitespace word → a ▁-prefixed word-start piece."""
    return ["▁" + w.lower().rstrip(".") for w in text.split()]


class TestRoleTranslation:
    def test_wx_roles_map_to_shabdabodha_ids(self) -> None:
        assert wx_role_id("karwA") == SHABDABODHA_LABELS["karta"]
        assert wx_role_id("karma") == SHABDABODHA_LABELS["karma"]
        assert wx_role_id("karaNam") == SHABDABODHA_LABELS["karana"]
        assert wx_role_id("sampraxAnam") == SHABDABODHA_LABELS["sampradana"]
        assert wx_role_id("apAxAnam") == SHABDABODHA_LABELS["apadana"]
        assert wx_role_id("aXikaraNam") == SHABDABODHA_LABELS["adhikarana"]
        assert wx_role_id("kriyA") == SHABDABODHA_LABELS["kriya"]
        assert wx_role_id("separator") == SHABDABODHA_LABELS["separator"]

    def test_unknown_role_maps_to_none(self) -> None:
        assert wx_role_id("notarole") == SHABDABODHA_LABELS["none"]

    def test_every_realizer_role_is_translatable(self) -> None:
        # Every WX role the realizer can emit has a śābdabodha mapping.
        for wx in WX_TO_SHABDABODHA:
            assert WX_TO_SHABDABODHA[wx] in SHABDABODHA_LABELS


class TestSentenceRoleIds:
    def test_one_id_per_piece(self) -> None:
        pieces = ["▁the", "▁man", "▁eats", "▁the", "▁fruit", "."]
        word_roles = [
            ("the", "separator"),
            ("man", "karwA"),
            ("eats", "kriyA"),
            ("the", "separator"),
            ("fruit", "karma"),
        ]
        ids = sentence_role_ids(pieces, word_roles)
        assert len(ids) == len(pieces)

    def test_continuation_and_punct_are_separator(self) -> None:
        # "eats" splits into ▁eat + s; the role lands on the first piece, the
        # continuation "s" and the trailing "." are separator.
        pieces = ["▁the", "▁man", "▁eat", "s", "▁the", "▁fruit", "."]
        word_roles = [
            ("the", "separator"),
            ("man", "karwA"),
            ("eats", "kriyA"),
            ("the", "separator"),
            ("fruit", "karma"),
        ]
        sep = SHABDABODHA_LABELS["separator"]
        assert sentence_role_ids(pieces, word_roles) == [
            sep,
            SHABDABODHA_LABELS["karta"],
            SHABDABODHA_LABELS["kriya"],
            sep,  # "s" continuation
            sep,
            SHABDABODHA_LABELS["karma"],
            sep,  # "." punctuation
        ]


class TestBuildCorpus:
    def test_lines_and_roles_align_with_eos(self) -> None:
        frames = list(enumerate_frames(20, seed=0))
        lines, roles = build_contrast_corpus(frames, _fake_pieces, with_eos_role=True)
        assert len(lines) > 0
        # Alignment invariant (mirrors build_shabdabodha_cache): total role ids =
        # total tokens + one eos-role per line.
        total_tokens = sum(len(_fake_pieces(ln)) for ln in lines)
        assert len(roles) == total_tokens + len(lines)
        assert all(0 <= r < len(SHABDABODHA_LABELS) for r in roles)

    def test_active_and_passive_both_emitted(self) -> None:
        # A transitive frame contributes both voices (2 lines).
        frame = KarakaFrame(
            {
                "words": [
                    {
                        "id": 1,
                        "pos": "noun",
                        "stem": "nara",
                        "gender": "puM",
                        "number": "eka",
                        "karaka": "karwA",
                        "head": 3,
                    },
                    {
                        "id": 2,
                        "pos": "noun",
                        "stem": "Pala",
                        "gender": "napuM",
                        "number": "eka",
                        "karaka": "karma",
                        "head": 3,
                    },
                    {
                        "id": 3,
                        "pos": "verb",
                        "dhatu": "KAx1",
                        "prayoga": "karwari",
                        "lakara": "varwamAnaH",
                    },
                ]
            },
            ("t",),
        )
        lines, _ = build_contrast_corpus([frame], _fake_pieces)
        assert lines == ["The man eats the fruit.", "The fruit is eaten by the man."]


class TestShuffleControl:
    def test_shuffle_preserves_distribution_changes_order(self) -> None:
        roles = [0, 1, 2, 7, 8, 8, 1, 0, 9, 3, 8, 8] * 5
        shuffled = shuffle_roles(roles, seed=0)
        assert sorted(shuffled) == sorted(roles)  # same multiset
        assert shuffled != roles  # alignment destroyed


class TestRealTokenizerAlignment:
    """The confound-removal claim, measured: gold sentences align cleanly to the
    committed SentencePiece model (~0% residual, unlike the spaCy path)."""

    def test_gold_roles_land_on_the_right_pieces(self) -> None:
        spm = pytest.importorskip("sentencepiece")
        if not _SPM.exists():
            pytest.skip(f"tokenizer not present: {_SPM}")
        sp = spm.SentencePieceProcessor()
        sp.Load(str(_SPM))
        realizer = EnglishFrameRealizer()
        frame = _instrument_frame()
        sep = SHABDABODHA_LABELS["separator"]
        for voice in ("active", "passive"):
            sentence = realizer.realize(frame, voice=voice)
            word_roles = realizer.word_roles(frame, voice=voice)
            assert sentence is not None and word_roles is not None
            pieces = sp.EncodeAsPieces(sentence.text)
            # Clean alignment: exactly one word-start piece per gold word.
            word_starts = [p for p in pieces if p.startswith("▁")]
            assert len(word_starts) == len(word_roles)
            ids = sentence_role_ids(pieces, word_roles)
            assert len(ids) == len(pieces)
            # The non-separator ids equal, in order, the gold head roles —
            # every kāraka landed on its head piece, nothing drifted.
            got_heads = [i for i in ids if i != sep]
            gold_heads = [wx_role_id(r) for _, r in word_roles if r != "separator"]
            assert got_heads == gold_heads
