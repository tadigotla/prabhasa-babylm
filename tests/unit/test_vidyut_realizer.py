"""Correctness + acceptance tests for the Vidyut-native frame realizer (ADR-0033).

These run only when the optional ``vidyut`` wheel is installed (aarch64 host /
GB10 container); they are the standalone-closure evidence for the
``vidyut-realize`` workstream. Every assertion checks a *real* classical Sanskrit
form — no mocks, no fabricated surfaces.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.vidyut

pytest.importorskip("vidyut")

from psalm.domain.data.karaka_frames import KarakaFrame, enumerate_frames  # noqa: E402
from psalm.infrastructure.generators.vidyut_realizer import (  # noqa: E402
    VidyutFrameRealizer,
    VidyutRealizerConfig,
    forward_sandhi,
    frame_transitivity_violation,
)


def _noun(idx, stem, gender, number, karaka, head):
    return {
        "id": idx,
        "pos": "noun",
        "stem": stem,
        "gender": gender,
        "number": number,
        "karaka": karaka,
        "head": head,
    }


def _verb(idx, dhatu, lakara):
    return {"id": idx, "pos": "verb", "dhatu": dhatu, "prayoga": "karwari", "lakara": lakara}


@pytest.fixture(scope="module")
def realizer() -> VidyutFrameRealizer:
    return VidyutFrameRealizer()


# --- declension correctness (kāraka → vibhakti) ------------------------------


@pytest.mark.parametrize(
    ("stem", "gender", "karaka", "expected"),
    [
        ("nara", "puM", "karwA", "naraH"),  # masc a-stem nominative
        ("nara", "puM", "karma", "naram"),  # accusative
        ("nara", "puM", "karaNam", "nareRa"),  # instrumental
        ("nara", "puM", "sampraxAnam", "narAya"),  # dative
        ("nara", "puM", "aXikaraNam", "nare"),  # locative
        ("vana", "napuM", "karma", "vanam"),  # neuter acc
        ("siwA", "swrI", "karwA", "sitA"),  # fem ā-stem nom (needs nyāp)
        ("naxI", "swrI", "karwA", "nadI"),  # fem ī-stem nom (needs nyāp)
        ("guru", "puM", "karma", "gurum"),  # masc i-stem acc
    ],
)
def test_declension_matches_classical(realizer, stem, gender, karaka, expected) -> None:
    surface = realizer._decline(_noun(1, stem, gender, "eka", karaka, 2))
    assert surface == expected


# --- conjugation correctness (dhātu → tiṅanta) -------------------------------


@pytest.mark.parametrize(
    ("dhatu", "expected"),
    [
        ("gam1", "gacCati"),  # gam → gacchati
        ("paW1", "paWati"),  # paṭh → paṭhati
        ("KAx1", "KAdati"),  # khād → khādati
        ("xfS1", "paSyati"),  # dṛś → paśyati
        ("BU1", "Bavati"),  # bhū → bhavati
        ("vas1", "vasati"),  # vas → vasati
    ],
)
def test_conjugation_present_3sg(realizer, dhatu, expected) -> None:
    surface, derivation = realizer._conjugate(_verb(9, dhatu, "varwamAnaH"), "eka")
    assert surface == expected
    assert derivation  # sūtra-by-sūtra gold derivation is non-empty


def test_conjugation_vacana_from_karta(realizer) -> None:
    # plural kartā → plural verb
    sg, _ = realizer._conjugate(_verb(9, "gam1", "varwamAnaH"), "eka")
    pl, _ = realizer._conjugate(_verb(9, "gam1", "varwamAnaH"), "bahu")
    assert sg == "gacCati"
    assert pl == "gacCanti"


# --- full-frame realization with gold kāraka ---------------------------------


def test_realize_transitive_frame_gold_parse(realizer) -> None:
    frame = KarakaFrame(
        {
            "words": [
                _noun(1, "nara", "puM", "eka", "karwA", 3),
                _noun(2, "Pala", "napuM", "eka", "karma", 3),
                _verb(3, "KAx1", "varwamAnaH"),
            ]
        },
        signature=("KAx1", "varwamAnaH", "nara", "eka", "Pala", "eka"),
    )
    out = realizer.realize(frame)
    assert out is not None
    assert out.karaka_parse == (("naraH", "karwA"), ("Palam", "karma"), ("KAdati", "kriyA"))
    # every gold token appears in the surface text (lossless w.r.t. padas)
    for token, _role in out.karaka_parse:
        assert token in out.text
    assert out.meta["scheme"] == "slp1"
    assert out.meta["sandhi"] in {"full", "partial", "none"}
    assert out.has_gold_parse


def test_realize_intransitive_frame(realizer) -> None:
    frame = KarakaFrame(
        {
            "words": [
                _noun(1, "bAla", "puM", "eka", "karwA", 2),
                _verb(2, "gam1", "varwamAnaH"),
            ]
        },
        signature=("gam1", "varwamAnaH", "bAla", "eka", "-", "-"),
    )
    out = realizer.realize(frame)
    assert out is not None
    assert out.karaka_parse == (("bAlaH", "karwA"), ("gacCati", "kriyA"))


# --- transitivity guard (akarmaka never takes karma) -------------------------


def test_transitivity_violation_detected_and_skipped(realizer) -> None:
    bad = KarakaFrame(
        {
            "words": [
                _noun(1, "nara", "puM", "eka", "karwA", 3),
                _noun(2, "Pala", "napuM", "eka", "karma", 3),
                _verb(3, "BU1", "varwamAnaH"),  # bhū is akarmaka → karma illegal
            ]
        },
        signature=("BU1", "varwamAnaH", "nara", "eka", "Pala", "eka"),
    )
    assert frame_transitivity_violation(bad) is True
    assert realizer.realize(bad) is None


def test_enumerated_frames_have_no_transitivity_violations() -> None:
    frames = list(enumerate_frames(3000, seed=0))
    violations = [f for f in frames if frame_transitivity_violation(f)]
    assert violations == [], f"{len(violations)} akarmaka+karma frames leaked"


# --- karmaṇi passive (Sanskrit side of the H1_CONTRAST role contrast) ----------


def _transitive_frame(karma_number: str = "eka") -> KarakaFrame:
    return KarakaFrame(
        {
            "words": [
                _noun(1, "nara", "puM", "eka", "karwA", 3),
                _noun(2, "Pala", "napuM", karma_number, "karma", 3),
                _verb(3, "KAx1", "varwamAnaH"),
            ]
        },
        signature=("KAx1", "varwamAnaH", "nara", "eka", "Pala", karma_number),
    )


def test_realize_passive_transitive_gold_parse(realizer) -> None:
    out = realizer.realize(_transitive_frame(), voice="passive")
    assert out is not None
    # kartā → instrumental (nareRa), karma → nominative (Palam), verb → karmaṇi.
    # The gold roles are identical to the active frame; only the surface moves.
    assert out.karaka_parse == (
        ("nareRa", "karwA"),
        ("Palam", "karma"),
        ("KAdyate", "kriyA"),
    )
    assert out.meta["prayoga"] == "karmani"
    for token, _role in out.karaka_parse:
        assert token in out.text


def test_passive_verb_agrees_with_patient_number(realizer) -> None:
    out = realizer.realize(_transitive_frame(karma_number="bahu"), voice="passive")
    assert out is not None
    # Passive verb agrees with the patient (the new nominative subject) → plural.
    assert "KAdyante" in out.text


def test_passive_intransitive_returns_none(realizer) -> None:
    frame = KarakaFrame(
        {
            "words": [
                _noun(1, "bAla", "puM", "eka", "karwA", 2),
                _verb(2, "gam1", "varwamAnaH"),
            ]
        },
        signature=("gam1", "varwamAnaH", "bAla", "eka", "-", "-"),
    )
    # No karma → no karmaṇi passive.
    assert realizer.realize(frame, voice="passive") is None


def test_active_is_unchanged_default(realizer) -> None:
    # Regression guard: default voice stays active and identical to before.
    out = realizer.realize(_transitive_frame())
    assert out is not None
    assert out.karaka_parse == (("naraH", "karwA"), ("Palam", "karma"), ("KAdati", "kriyA"))
    assert out.meta["prayoga"] == "karwari"


# --- forward sandhi honesty (no fabricated fusions) --------------------------


def test_forward_sandhi_vowel_rules() -> None:
    assert forward_sandhi(["rAma", "iti"]) == ("rAmeti", "full")  # a+i→e
    assert forward_sandhi(["na", "asti"]) == ("nAsti", "full")  # a+a→A
    assert forward_sandhi(["sA", "uvAca"]) == ("sovAca", "full")  # ā+u→o


def test_forward_sandhi_partial_on_consonant_junction() -> None:
    text, mode = forward_sandhi(["naraH", "gacCati"])
    assert text == "naraH gacCati"  # visarga junction left as boundary (padapāṭha)
    assert mode == "partial"


def test_forward_sandhi_single_and_empty() -> None:
    assert forward_sandhi(["naraH"]) == ("naraH", "none")
    assert forward_sandhi([]) == ("", "none")


# --- acceptance: stream realizes ~100% with gold parse -----------------------


def test_stream_acceptance_rate_and_gold(realizer) -> None:
    n = 1500
    items = list(realizer.stream(n, seed=0))
    assert len(items) == n
    assert all(s.has_gold_parse for s in items)
    # every example carries scheme + sandhi metadata and a real (non-label) surface
    for s in items:
        assert s.meta["generator"] == "vidyut-realizer"
        assert "." not in s.text  # not the old pseudo-surface label form
        assert s.language == "sa"


def test_stream_realization_success_fraction(realizer) -> None:
    frames = list(enumerate_frames(1000, seed=1))
    realized = [realizer.realize(f) for f in frames]
    ok = [r for r in realized if r is not None]
    assert len(ok) / len(frames) >= 0.98


# --- edge paths: None returns, config toggles, stream bounds -----------------


def test_unknown_stem_returns_none(realizer) -> None:
    assert realizer._decline(_noun(1, "NOT_A_STEM", "puM", "eka", "karwA", 2)) is None


def test_unknown_karaka_returns_none(realizer) -> None:
    assert realizer._decline(_noun(1, "nara", "puM", "eka", "hetuH", 2)) is None


def test_unknown_dhatu_and_lakara_return_none(realizer) -> None:
    assert realizer._conjugate(_verb(9, "NOPE9", "varwamAnaH"), "eka") == (None, ())
    assert realizer._conjugate(_verb(9, "gam1", "NOPE"), "eka") == (None, ())


def test_realize_without_sandhi_space_joins() -> None:
    r = VidyutFrameRealizer(VidyutRealizerConfig(apply_sandhi=False, include_derivation=False))
    frame = KarakaFrame(
        {
            "words": [
                _noun(1, "nara", "puM", "eka", "karwA", 3),
                _noun(2, "Pala", "napuM", "eka", "karma", 3),
                _verb(3, "KAx1", "varwamAnaH"),
            ]
        },
        signature=("KAx1", "varwamAnaH", "nara", "eka", "Pala", "eka"),
    )
    out = r.realize(frame)
    assert out is not None
    assert out.text == "naraH Palam KAdati"
    assert out.meta["sandhi"] == "none"
    assert out.derivation == ()  # derivation disabled


def test_stream_bounds(realizer) -> None:
    assert list(realizer.stream(0)) == []
    with pytest.raises(ValueError):
        list(realizer.stream(-1))
