"""SA↔EN bilingual role-set cross-check (task 4.3 of H1_CONTRAST, ADR-0043).

The same :class:`KarakaFrame` realized by the Vidyut Sanskrit realizer and by the
English frame realizer must carry an identical gold kāraka role inventory. That
is the property the contrast supervision rests on: one meaning, two surfaces, one
role set — and it holds across the active↔passive voice contrast in both
languages. Requires the optional Vidyut wheel.

A nice illustration falls out of the passive: in Sanskrit the agent and the
instrument both take the instrumental case (Tṛtīyā), yet their kāraka roles stay
distinct — exactly why kāraka (deep) is not vibhakti (surface). English keeps
them apart with "by" vs "with". The role inventory matches regardless.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.vidyut

pytest.importorskip("vidyut")

from psalm.application.data.ports import AnnotatedSentence  # noqa: E402
from psalm.domain.data.karaka_frames import KarakaFrame, enumerate_frames  # noqa: E402
from psalm.infrastructure.generators.english_frame_realizer import (  # noqa: E402
    EnglishFrameRealizer,
)
from psalm.infrastructure.generators.vidyut_realizer import VidyutFrameRealizer  # noqa: E402


def _roles(sentence: AnnotatedSentence) -> list[str]:
    return sorted(role for _, role in sentence.karaka_parse)


@pytest.fixture(scope="module")
def sa() -> VidyutFrameRealizer:
    return VidyutFrameRealizer()


@pytest.fixture(scope="module")
def en() -> EnglishFrameRealizer:
    return EnglishFrameRealizer()


def _frame() -> KarakaFrame:
    """nara (agent) eats Pala (patient) with puswaka (instrument)."""
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
        ("bilingual",),
    )


@pytest.mark.parametrize("voice", ["active", "passive"])
def test_sa_en_share_role_inventory(
    sa: VidyutFrameRealizer, en: EnglishFrameRealizer, voice: str
) -> None:
    frame = _frame()
    sa_out = sa.realize(frame, voice=voice)
    en_out = en.realize(frame, voice=voice)
    assert sa_out is not None and en_out is not None
    expected = ["karaNam", "karma", "karwA", "kriyA"]
    assert _roles(sa_out) == _roles(en_out) == expected


def test_role_inventory_is_voice_and_language_invariant(
    sa: VidyutFrameRealizer, en: EnglishFrameRealizer
) -> None:
    frame = _frame()
    outs = [
        sa.realize(frame, voice="active"),
        sa.realize(frame, voice="passive"),
        en.realize(frame, voice="active"),
        en.realize(frame, voice="passive"),
    ]
    assert all(o is not None for o in outs)
    inventories = {tuple(_roles(o)) for o in outs if o is not None}
    assert inventories == {("karaNam", "karma", "karwA", "kriyA")}


def test_cross_language_role_match_over_stream(
    sa: VidyutFrameRealizer, en: EnglishFrameRealizer
) -> None:
    checked = 0
    for frame in enumerate_frames(50, seed=0):
        sa_out = sa.realize(frame, voice="active")
        en_out = en.realize(frame, voice="active")
        assert sa_out is not None and en_out is not None
        assert _roles(sa_out) == _roles(en_out)
        checked += 1
    assert checked == 50
