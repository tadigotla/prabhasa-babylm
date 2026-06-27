"""Unit tests for the kāraka ↔ vibhakti ↔ English construction index (task 3.1).

The index is the single source of truth for how each kāraka (semantic role)
surfaces: which Sanskrit case (vibhakti) it takes in a given voice, and how it
realizes in English. The linguistically load-bearing cells are the voice shift —
kartā moves nominative→instrumental and karma moves accusative→nominative under
the passive — which is the deep/surface decoupling the whole mechanism rests on.
"""

from __future__ import annotations

import logging

import pytest

from psalm.domain.linguistics.construction_index import (
    ACTIVE,
    ALL_KARAKAS,
    PASSIVE,
    collapse_warning,
    construction_for,
    english_collapses,
    log_coverage,
    realized_constructions,
)


class TestActiveCells:
    def test_karta_is_nominative_subject(self) -> None:
        c = construction_for("karwA", ACTIVE)
        assert c is not None
        assert (c.vibhakti, c.slot, c.preposition) == ("Prathama", "subject", "")

    def test_karma_is_accusative_object(self) -> None:
        c = construction_for("karma", ACTIVE)
        assert c is not None
        assert (c.vibhakti, c.slot, c.preposition) == ("Dvitiya", "object", "")


class TestPassiveVoiceShift:
    def test_karta_becomes_instrumental_by_phrase(self) -> None:
        c = construction_for("karwA", PASSIVE)
        assert c is not None
        # The crux: agent's case shifts nominative → instrumental under passive.
        assert (c.vibhakti, c.slot, c.preposition) == ("Trtiya", "agent", "by ")

    def test_karma_becomes_nominative_subject(self) -> None:
        c = construction_for("karma", PASSIVE)
        assert c is not None
        assert (c.vibhakti, c.slot, c.preposition) == ("Prathama", "subject", "")


class TestObliqueCellsAreVoiceInvariant:
    def test_oblique_cases_and_prepositions(self) -> None:
        expected = {
            "karaNam": ("Trtiya", "with "),
            "sampraxAnam": ("Caturthi", "to "),
            "apAxAnam": ("Panchami", "from "),
            "aXikaraNam": ("Saptami", "in "),
        }
        for role, (vibhakti, prep) in expected.items():
            for voice in (ACTIVE, PASSIVE):
                c = construction_for(role, voice)
                assert c is not None, f"{role}/{voice} missing"
                assert c.vibhakti == vibhakti
                assert c.preposition == prep
                assert c.slot == "oblique"


class TestCoverageAndGuards:
    def test_unknown_role_returns_none(self) -> None:
        assert construction_for("notakaraka", ACTIVE) is None

    def test_all_six_karakas_realize_in_both_voices(self) -> None:
        for role in ALL_KARAKAS:
            assert construction_for(role, ACTIVE) is not None
            assert construction_for(role, PASSIVE) is not None


class TestEnglishCollapseAudit:
    def test_six_karaka_set_has_no_english_collapse_per_voice(self) -> None:
        # Honest property: at full-NP granularity, English distinguishes all six
        # kārakas within a single voice — nothing to drop.
        for voice in (ACTIVE, PASSIVE):
            cons = [construction_for(r, voice) for r in ALL_KARAKAS]
            assert english_collapses([c for c in cons if c is not None]) == []

    def test_audit_catches_a_synthetic_with_collapse(self) -> None:
        # If a finer role (e.g. a comitative also surfacing as "with") were added
        # alongside the instrument, the audit must flag the collapse, not let it
        # be silently mislabeled.
        instrument = construction_for("karaNam", ACTIVE)
        assert instrument is not None
        comitative = instrument.__class__(
            karaka="sahArtha",  # "togetherness" — not a kāraka; also "with"
            voice=ACTIVE,
            vibhakti="Trtiya",
            slot="oblique",
            preposition="with ",
        )
        groups = english_collapses([instrument, comitative])
        assert len(groups) == 1
        assert {c.karaka for c in groups[0]} == {"karaNam", "sahArtha"}


class TestCoverageLogging:
    def test_realized_constructions_cover_all_six_in_both_voices(self) -> None:
        for voice in (ACTIVE, PASSIVE):
            assert {c.karaka for c in realized_constructions(voice)} == set(ALL_KARAKAS)

    def test_collapse_warning_names_members_and_drop(self) -> None:
        instrument = construction_for("karaNam", ACTIVE)
        assert instrument is not None
        comitative = instrument.__class__(
            karaka="sahArtha",
            voice=ACTIVE,
            vibhakti="Trtiya",
            slot="oblique",
            preposition="with ",
        )
        msg = collapse_warning([instrument, comitative])
        assert "karaNam" in msg and "sahArtha" in msg
        assert "dropping" in msg

    def test_log_coverage_reports_full_coverage_no_collapse(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            log_coverage()
        assert "6 kārakas realized, 0 collapse-group(s)" in caplog.text
        # Honest property: nothing dropped → no collapse warnings emitted.
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
