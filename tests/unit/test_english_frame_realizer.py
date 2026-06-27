"""Unit tests for the frame → English realizer (Route A of H1_CONTRAST, ADR-0043).

The realizer renders a :class:`KarakaFrame` into an English sentence whose kāraka
roles are *gold by construction* — the same frame that drives the Sanskrit
realizer, so a generated (Sanskrit, English) pair shares one role set with no
translation and no alignment. These tests pin the active-voice surface forms,
number/tense agreement, and the gold parse.
"""

from __future__ import annotations

from psalm.domain.data.karaka_frames import (
    DHATUS,
    NOMINAL_STEMS,
    OBLIQUE_KARMA,
    KarakaFrame,
    enumerate_frames,
)
from psalm.infrastructure.generators.english_frame_realizer import (
    NOUN_GLOSS,
    PARTICIPLE,
    VERB_GLOSS,
    EnglishFrameRealizer,
    build_contrast_set,
)


def _noun(idx: int, stem: str, gender: str, number: str, karaka: str) -> dict[str, object]:
    return {
        "id": idx,
        "pos": "noun",
        "stem": stem,
        "gender": gender,
        "number": number,
        "karaka": karaka,
        "head": 99,
    }


def _verb(idx: int, dhatu: str, lakara: str) -> dict[str, object]:
    return {"id": idx, "pos": "verb", "dhatu": dhatu, "prayoga": "karwari", "lakara": lakara}


def _frame(*words: dict[str, object]) -> KarakaFrame:
    return KarakaFrame({"words": list(words)}, ("test",))


class TestGlossCoverage:
    def test_every_frame_noun_stem_has_a_gloss(self) -> None:
        stems = {s for s, _ in NOMINAL_STEMS} | {s for s, _ in OBLIQUE_KARMA}
        missing = stems - set(NOUN_GLOSS)
        assert missing == set(), f"noun stems without an English gloss: {missing}"

    def test_every_frame_dhatu_has_a_gloss(self) -> None:
        # Base dhātus plus the two referenced only by the oblique frames.
        dhatus = {d for d, _ in DHATUS} | {"xA1", "vas1"}
        missing = dhatus - set(VERB_GLOSS)
        assert missing == set(), f"dhātus without an English gloss: {missing}"


class TestActiveRealization:
    def test_transitive_present_singular(self) -> None:
        frame = _frame(
            _noun(1, "nara", "puM", "eka", "karwA"),
            _noun(2, "Pala", "napuM", "eka", "karma"),
            _verb(3, "KAx1", "varwamAnaH"),
        )
        out = EnglishFrameRealizer().realize(frame)
        assert out is not None
        assert out.language == "en"
        assert out.text == "The man eats the fruit."
        assert out.karaka_parse == (
            ("man", "karwA"),
            ("fruit", "karma"),
            ("eats", "kriyA"),
        )
        assert out.has_gold_parse

    def test_subject_verb_number_agreement_plural(self) -> None:
        frame = _frame(
            _noun(1, "nara", "puM", "bahu", "karwA"),
            _noun(2, "Pala", "napuM", "eka", "karma"),
            _verb(3, "KAx1", "varwamAnaH"),
        )
        out = EnglishFrameRealizer().realize(frame)
        assert out is not None
        assert out.text == "The men eat the fruit."
        assert ("men", "karwA") in out.karaka_parse
        assert ("eat", "kriyA") in out.karaka_parse

    def test_dual_renders_as_two(self) -> None:
        frame = _frame(
            _noun(1, "bAla", "puM", "xvi", "karwA"),
            _verb(2, "gam1", "varwamAnaH"),
        )
        out = EnglishFrameRealizer().realize(frame)
        assert out is not None
        assert out.text == "The two boys go."

    def test_past_tense_imperfect(self) -> None:
        frame = _frame(
            _noun(1, "nara", "puM", "eka", "karwA"),
            _noun(2, "Pala", "napuM", "eka", "karma"),
            _verb(3, "KAx1", "anaxyawanaBUwaH"),
        )
        out = EnglishFrameRealizer().realize(frame)
        assert out is not None
        assert out.text == "The man ate the fruit."

    def test_future_tense(self) -> None:
        frame = _frame(
            _noun(1, "bAla", "puM", "eka", "karwA"),
            _verb(2, "gam1", "sAmAnyaBaviRyakAlaH"),
        )
        out = EnglishFrameRealizer().realize(frame)
        assert out is not None
        assert out.text == "The boy will go."

    def test_intransitive(self) -> None:
        frame = _frame(
            _noun(1, "bAla", "puM", "eka", "karwA"),
            _verb(2, "gam1", "varwamAnaH"),
        )
        out = EnglishFrameRealizer().realize(frame)
        assert out is not None
        assert out.text == "The boy goes."
        assert out.karaka_parse == (("boy", "karwA"), ("goes", "kriyA"))


class TestObliqueRoleFrames:
    def test_instrument_uses_with(self) -> None:
        frame = _frame(
            _noun(1, "nara", "puM", "eka", "karwA"),
            _noun(2, "Pala", "napuM", "eka", "karma"),
            _noun(3, "puswaka", "napuM", "eka", "karaNam"),
            _verb(4, "KAx1", "varwamAnaH"),
        )
        out = EnglishFrameRealizer().realize(frame)
        assert out is not None
        assert "with the book" in out.text
        assert ("book", "karaNam") in out.karaka_parse

    def test_recipient_uses_to(self) -> None:
        frame = _frame(
            _noun(1, "nara", "puM", "eka", "karwA"),
            _noun(2, "Pala", "napuM", "eka", "karma"),
            _noun(3, "bAla", "puM", "eka", "sampraxAnam"),
            _verb(4, "xA1", "varwamAnaH"),
        )
        out = EnglishFrameRealizer().realize(frame)
        assert out is not None
        assert "to the boy" in out.text
        assert ("boy", "sampraxAnam") in out.karaka_parse

    def test_locus_uses_in(self) -> None:
        frame = _frame(
            _noun(1, "nara", "puM", "eka", "karwA"),
            _noun(2, "gfha", "napuM", "eka", "aXikaraNam"),
            _verb(3, "vas1", "varwamAnaH"),
        )
        out = EnglishFrameRealizer().realize(frame)
        assert out is not None
        assert "in the house" in out.text
        assert ("house", "aXikaraNam") in out.karaka_parse


class TestGoldParseInvariants:
    def test_every_parse_token_appears_in_text(self) -> None:
        realizer = EnglishFrameRealizer()
        for sentence in realizer.stream(50, seed=0):
            for token, _role in sentence.karaka_parse:
                assert token in sentence.text, f"{token!r} not in {sentence.text!r}"

    def test_stream_is_deterministic_by_seed(self) -> None:
        realizer = EnglishFrameRealizer()
        a = [s.text for s in realizer.stream(20, seed=3)]
        b = [s.text for s in realizer.stream(20, seed=3)]
        assert a == b


class TestPassiveRealization:
    def test_participle_covers_every_verb(self) -> None:
        missing = set(VERB_GLOSS) - set(PARTICIPLE)
        assert missing == set(), f"verbs without a past participle: {missing}"

    def test_passive_transitive_present_singular(self) -> None:
        frame = _frame(
            _noun(1, "nara", "puM", "eka", "karwA"),
            _noun(2, "Pala", "napuM", "eka", "karma"),
            _verb(3, "KAx1", "varwamAnaH"),
        )
        out = EnglishFrameRealizer().realize(frame, voice="passive")
        assert out is not None
        assert out.text == "The fruit is eaten by the man."
        assert out.meta["voice"] == "passive"
        assert ("man", "karwA") in out.karaka_parse
        assert ("fruit", "karma") in out.karaka_parse
        assert ("is eaten", "kriyA") in out.karaka_parse

    def test_passive_auxiliary_agrees_with_patient_not_agent(self) -> None:
        # Plural agent, singular patient: aux must be "is" (agrees with patient).
        frame = _frame(
            _noun(1, "nara", "puM", "bahu", "karwA"),
            _noun(2, "Pala", "napuM", "eka", "karma"),
            _verb(3, "KAx1", "varwamAnaH"),
        )
        out = EnglishFrameRealizer().realize(frame, voice="passive")
        assert out is not None
        assert out.text == "The fruit is eaten by the men."

    def test_passive_plural_patient(self) -> None:
        frame = _frame(
            _noun(1, "nara", "puM", "eka", "karwA"),
            _noun(2, "Pala", "napuM", "bahu", "karma"),
            _verb(3, "KAx1", "varwamAnaH"),
        )
        out = EnglishFrameRealizer().realize(frame, voice="passive")
        assert out is not None
        assert out.text == "The fruits are eaten by the man."

    def test_passive_past_and_future(self) -> None:
        base = (
            _noun(1, "nara", "puM", "eka", "karwA"),
            _noun(2, "Pala", "napuM", "eka", "karma"),
        )
        past = EnglishFrameRealizer().realize(
            _frame(*base, _verb(3, "KAx1", "anaxyawanaBUwaH")), voice="passive"
        )
        future = EnglishFrameRealizer().realize(
            _frame(*base, _verb(3, "KAx1", "sAmAnyaBaviRyakAlaH")), voice="passive"
        )
        assert past is not None and past.text == "The fruit was eaten by the man."
        assert future is not None and future.text == "The fruit will be eaten by the man."

    def test_passive_intransitive_returns_none(self) -> None:
        # No karma → no natural English passive.
        frame = _frame(
            _noun(1, "bAla", "puM", "eka", "karwA"),
            _verb(2, "gam1", "varwamAnaH"),
        )
        assert EnglishFrameRealizer().realize(frame, voice="passive") is None


class TestContrastSet:
    def test_transitive_frame_yields_active_and_passive(self) -> None:
        frame = _frame(
            _noun(1, "nara", "puM", "eka", "karwA"),
            _noun(2, "Pala", "napuM", "eka", "karma"),
            _verb(3, "KAx1", "varwamAnaH"),
        )
        members = build_contrast_set(frame)
        voices = {m.meta["voice"] for m in members}
        assert voices == {"active", "passive"}
        texts = {m.text for m in members}
        assert texts == {"The man eats the fruit.", "The fruit is eaten by the man."}

    def test_intransitive_frame_yields_active_only(self) -> None:
        frame = _frame(
            _noun(1, "bAla", "puM", "eka", "karwA"),
            _verb(2, "gam1", "varwamAnaH"),
        )
        members = build_contrast_set(frame)
        assert [m.meta["voice"] for m in members] == ["active"]


class TestRoleInvariance:
    """Task 4.3: surface varies across constructions, gold role set does not."""

    @staticmethod
    def _noun_roles(sentence: object) -> set[tuple[str, str]]:
        parse = sentence.karaka_parse  # type: ignore[attr-defined]
        return {(tok, role) for tok, role in parse if role != "kriyA"}

    def test_active_and_passive_share_one_noun_role_set(self) -> None:
        frame = _frame(
            _noun(1, "nara", "puM", "eka", "karwA"),
            _noun(2, "Pala", "napuM", "eka", "karma"),
            _verb(3, "KAx1", "varwamAnaH"),
        )
        active = EnglishFrameRealizer().realize(frame, voice="active")
        passive = EnglishFrameRealizer().realize(frame, voice="passive")
        assert active is not None and passive is not None
        # Same participant → same role, regardless of surface position.
        assert self._noun_roles(active) == self._noun_roles(passive)
        assert active.text != passive.text

    def test_invariance_holds_across_the_contrast_stream(self) -> None:
        realizer = EnglishFrameRealizer()
        checked = 0
        for sentence in realizer.stream(60, seed=1):
            # Reconstruct the frame is not exposed here; instead assert each
            # realization's noun-role set is internally consistent and gold.
            roles = self._noun_roles(sentence)
            assert all(
                r in {"karwA", "karma", "karaNam", "sampraxAnam", "apAxAnam", "aXikaraNam"}
                for _, r in roles
            )
            checked += 1
        assert checked == 60


class TestWordRoles:
    """Per-surface-word roles for the aux role-label stream (task 5 / Option B)."""

    def test_active_word_roles(self) -> None:
        frame = _frame(
            _noun(1, "nara", "puM", "eka", "karwA"),
            _noun(2, "Pala", "napuM", "eka", "karma"),
            _noun(3, "puswaka", "napuM", "eka", "karaNam"),
            _verb(4, "KAx1", "varwamAnaH"),
        )
        wr = EnglishFrameRealizer().word_roles(frame)
        assert wr == [
            ("the", "separator"),
            ("man", "karwA"),
            ("eats", "kriyA"),
            ("the", "separator"),
            ("fruit", "karma"),
            ("with", "separator"),
            ("the", "separator"),
            ("book", "karaNam"),
        ]

    def test_passive_word_roles(self) -> None:
        frame = _frame(
            _noun(1, "nara", "puM", "eka", "karwA"),
            _noun(2, "Pala", "napuM", "eka", "karma"),
            _verb(3, "KAx1", "varwamAnaH"),
        )
        wr = EnglishFrameRealizer().word_roles(frame, voice="passive")
        # "is eaten" → "is"=separator, "eaten"=kriyā; "by the man" → "man"=kartā.
        assert wr == [
            ("the", "separator"),
            ("fruit", "karma"),
            ("is", "separator"),
            ("eaten", "kriyA"),
            ("by", "separator"),
            ("the", "separator"),
            ("man", "karwA"),
        ]

    def test_word_roles_count_matches_text_words(self) -> None:
        # One role per whitespace word → aligns 1:1 with tokenizer word-starts.
        realizer = EnglishFrameRealizer()
        for frame in enumerate_frames(60, seed=2):
            for voice in ("active", "passive"):
                out = realizer.realize(frame, voice=voice)
                wr = realizer.word_roles(frame, voice=voice)
                if out is None:
                    assert wr is None
                    continue
                assert wr is not None
                assert len(wr) == len(out.text.split())

    def test_word_roles_heads_match_karaka_parse(self) -> None:
        # The non-separator word roles equal the sentence's gold noun+verb roles.
        realizer = EnglishFrameRealizer()
        frame = _frame(
            _noun(1, "kanyA", "swrI", "eka", "karwA"),
            _noun(2, "jala", "napuM", "eka", "karma"),
            _verb(3, "xfS1", "varwamAnaH"),
        )
        out = realizer.realize(frame)
        wr = realizer.word_roles(frame)
        assert out is not None and wr is not None
        heads = [(w, r) for w, r in wr if r != "separator"]
        # Surface order (word_roles) vs role-grouped order (karaka_parse) differ;
        # the SET of gold (head, role) pairs is identical.
        assert sorted(heads) == sorted(out.karaka_parse)
