"""Frame → English realizer (Route A of H1_CONTRAST, ADR-0043).

Renders a :class:`~psalm.domain.data.karaka_frames.KarakaFrame` — the same
meaning structure that drives :mod:`vidyut_realizer` for Sanskrit — into an
English sentence whose kāraka roles are **gold by construction**. Because both
realizers read one frame, a generated (Sanskrit, English) pair carries an
identical role set with no translation and no word alignment: the frame is the
interlingua.

Scope: active voice (kartari). Kāraka roles surface through English's own
devices — word order for kartā/karma (subject/object), prepositions for the
oblique kārakas (instrument "with", recipient "to", source "from", locus "in").
The lexicon is the frame lexicon; vocabulary breadth is bounded by it. Passive
(karmaṇi) and kṛt constructions are added separately.

``karaka_parse`` tags each content head word (and the verb, ``kriyA``) with its
WX role name — the same vocabulary the Sanskrit realizer emits — so the two
sides' role sets compare directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

from psalm.application.data.ports import AnnotatedSentence
from psalm.domain.data.karaka_frames import KarakaFrame, enumerate_frames
from psalm.domain.linguistics.construction_index import ACTIVE, PASSIVE, construction_for

Word = dict[str, object]

#: WX nominal stem → (singular, plural) English common-noun gloss. Glosses are
#: common nouns (article-taking, countable) so article/number handling is uniform.
NOUN_GLOSS: dict[str, tuple[str, str]] = {
    "rAma": ("king", "kings"),
    "nara": ("man", "men"),
    "bAla": ("boy", "boys"),
    "guru": ("teacher", "teachers"),
    "aSva": ("horse", "horses"),
    "vana": ("forest", "forests"),
    "Pala": ("fruit", "fruits"),
    "jala": ("water", "waters"),
    "puswaka": ("book", "books"),
    "gfha": ("house", "houses"),
    "siwA": ("queen", "queens"),
    "kanyA": ("girl", "girls"),
    "naxI": ("river", "rivers"),
    "vixyA": ("lesson", "lessons"),
}

#: WX dhātu → (base, present-3sg, present-plural, past-sg, past-pl) English forms.
#: Future is "will " + base; optative is "may " + base.
VERB_GLOSS: dict[str, tuple[str, str, str, str, str]] = {
    "gam1": ("go", "goes", "go", "went", "went"),
    "paW1": ("read", "reads", "read", "read", "read"),
    "KAx1": ("eat", "eats", "eat", "ate", "ate"),
    "xfS1": ("see", "sees", "see", "saw", "saw"),
    "kf8": ("make", "makes", "make", "made", "made"),
    "BU1": ("be", "is", "are", "was", "were"),
    "sWA1": ("stand", "stands", "stand", "stood", "stood"),
    "vax1": ("speak", "speaks", "speak", "spoke", "spoke"),
    "xA1": ("give", "gives", "give", "gave", "gave"),
    "vas1": ("dwell", "dwells", "dwell", "dwelt", "dwelt"),
}

#: WX dhātu → English past participle (for the karmaṇi/passive construction).
PARTICIPLE: dict[str, str] = {
    "gam1": "gone",
    "paW1": "read",
    "KAx1": "eaten",
    "xfS1": "seen",
    "kf8": "made",
    "BU1": "been",
    "sWA1": "stood",
    "vax1": "spoken",
    "xA1": "given",
    "vas1": "dwelt",
}

#: Fixed surface order for oblique kārakas (after subject-verb-object). The slot
#: and preposition for every kāraka come from the construction index, the single
#: source of truth for the kāraka ↔ vibhakti ↔ English mapping.
OBLIQUE_ORDER: tuple[str, ...] = ("karaNam", "sampraxAnam", "apAxAnam", "aXikaraNam")

#: WX lakāra → tense category used for verb form selection.
_PRESENT = "varwamAnaH"
_PAST = "anaxyawanaBUwaH"
_FUTURE = "sAmAnyaBaviRyakAlaH"
_OPTATIVE = "viXiH"


class EnglishFrameRealizer:
    """Realize :class:`KarakaFrame`s as English sentences with a gold role parse.

    Mirrors the :class:`SentenceGenerator` shape of the Sanskrit realizer:
    :meth:`realize` renders one frame; :meth:`stream` enumerates frames
    deterministically by ``seed``.
    """

    def _np(self, noun: Word) -> tuple[str, str] | None:
        """Return ``(noun_phrase, head_word)`` for a noun, or ``None`` if no gloss."""
        gloss = NOUN_GLOSS.get(str(noun["stem"]))
        if gloss is None:
            return None
        singular, plural = gloss
        number = str(noun["number"])
        if number == "eka":
            return f"the {singular}", singular
        if number == "xvi":
            return f"the two {plural}", plural
        return f"the {plural}", plural  # bahu

    def _verb_form(self, verb: Word, subj_plural: bool) -> str | None:
        """Return the inflected English verb form, or ``None`` if unglossed/untensed."""
        forms = VERB_GLOSS.get(str(verb["dhatu"]))
        if forms is None:
            return None
        base, pres_sg, pres_pl, past_sg, past_pl = forms
        lakara = str(verb.get("lakara", ""))
        if lakara == _PRESENT:
            return pres_pl if subj_plural else pres_sg
        if lakara == _PAST:
            return past_pl if subj_plural else past_sg
        if lakara == _FUTURE:
            return f"will {base}"
        if lakara == _OPTATIVE:
            return f"may {base}"
        return None

    def _prepare(
        self, frame: KarakaFrame
    ) -> tuple[dict[str, Word], dict[str, tuple[str, str]], Word] | None:
        """Parse a frame into (nouns-by-role, noun phrases, verb), or ``None``."""
        words = cast("list[Word]", frame.structure.get("words", []))
        nouns = [w for w in words if w.get("pos") == "noun"]
        verbs = [w for w in words if w.get("pos") == "verb"]
        if len(verbs) != 1:
            return None
        by_role: dict[str, Word] = {}
        for nw in nouns:
            by_role.setdefault(str(nw["karaka"]), nw)
        nps: dict[str, tuple[str, str]] = {}
        for role, nw in by_role.items():
            built = self._np(nw)
            if built is None:
                return None
            nps[role] = built
        return by_role, nps, verbs[0]

    @staticmethod
    def _finish(parts: list[str]) -> str:
        sentence = " ".join(parts)
        return f"{sentence[0].upper()}{sentence[1:]}."

    @staticmethod
    def _noun_parse(nps: dict[str, tuple[str, str]]) -> list[tuple[str, str]]:
        """Gold (head, role) pairs for the nouns, in canonical role order."""
        parse: list[tuple[str, str]] = []
        if "karwA" in nps:
            parse.append((nps["karwA"][1], "karwA"))
        if "karma" in nps:
            parse.append((nps["karma"][1], "karma"))
        for role in OBLIQUE_ORDER:
            if role in nps:
                parse.append((nps[role][1], role))
        return parse

    @staticmethod
    def _be(plural: bool, lakara: str) -> str | None:
        """The English auxiliary 'be' by number and tense, or ``None`` if untensed."""
        _, is_sg, are_pl, was_sg, were_pl = VERB_GLOSS["BU1"]
        if lakara == _PRESENT:
            return are_pl if plural else is_sg
        if lakara == _PAST:
            return were_pl if plural else was_sg
        if lakara == _FUTURE:
            return "will be"
        if lakara == _OPTATIVE:
            return "may be"
        return None

    @staticmethod
    def _meta(voice: str, verb: Word) -> dict[str, str]:
        return {
            "voice": voice,
            "lakara": str(verb.get("lakara", "")),
            "scheme": "en",
            "generator": "english-frame-realizer",
        }

    @staticmethod
    def _role_in_slot(nps: dict[str, tuple[str, str]], voice: str, slot: str) -> str | None:
        """Return the role occupying ``slot`` in ``voice``, per the index, or ``None``."""
        for role in nps:
            construction = construction_for(role, voice)
            if construction is not None and construction.slot == slot:
                return role
        return None

    def realize(self, frame: KarakaFrame, voice: str = "active") -> AnnotatedSentence | None:
        """Realize one frame as an English sentence in ``voice``, or ``None``.

        ``None`` is returned (never a fabricated string) when the frame lacks a
        single verb, a stem/dhātu is outside the gloss lexicon, the tense is
        unknown, or the requested voice does not apply (passive needs a karma).
        """
        prepared = self._prepare(frame)
        if prepared is None:
            return None
        by_role, nps, verb = prepared
        chunks = self._chunks(by_role, nps, verb, voice)
        if chunks is None:
            return None
        verb_form = next(text for text, role in chunks if role == "kriyA")
        parse = self._noun_parse(nps)
        parse.append((verb_form, "kriyA"))
        return AnnotatedSentence(
            text=self._finish([text for text, _ in chunks]),
            language="en",
            karaka_parse=tuple(parse),
            meta=self._meta(voice, verb),
        )

    def word_roles(self, frame: KarakaFrame, voice: str = "active") -> list[tuple[str, str]] | None:
        """Per-surface-word ``(word, role)`` sequence for the aux role-label stream.

        One entry per whitespace word, in surface order: the head of each chunk
        carries its kāraka (WX role name), every other word is ``separator``. The
        length equals the realized text's word count, so it aligns 1:1 with the
        tokenizer's word-start pieces (via ``align_pieces_to_role_ids``).
        """
        prepared = self._prepare(frame)
        if prepared is None:
            return None
        by_role, nps, verb = prepared
        chunks = self._chunks(by_role, nps, verb, voice)
        if chunks is None:
            return None
        out: list[tuple[str, str]] = []
        for text, role in chunks:
            words = text.split()
            for i, word in enumerate(words):
                out.append((word, role if i == len(words) - 1 else "separator"))
        return out

    def _chunks(
        self,
        by_role: dict[str, Word],
        nps: dict[str, tuple[str, str]],
        verb: Word,
        voice: str,
    ) -> list[tuple[str, str]] | None:
        """Surface-order ``(chunk, head-role)`` list shared by realize + word_roles.

        Each chunk's role is the kāraka of its head (the chunk's last word); the
        verb chunk's role is ``kriyA``. ``None`` if the voice does not apply.
        """
        if voice == "active":
            return self._active_chunks(by_role, nps, verb)
        if voice == "passive":
            return self._passive_chunks(by_role, nps, verb)
        return None

    def _oblique_part(self, role: str, voice: str, nps: dict[str, tuple[str, str]]) -> str | None:
        """Prepositional phrase for an oblique role, per the index, or ``None``."""
        construction = construction_for(role, voice)
        if construction is None:
            return None
        return construction.preposition + nps[role][0]

    def _active_chunks(
        self, by_role: dict[str, Word], nps: dict[str, tuple[str, str]], verb: Word
    ) -> list[tuple[str, str]] | None:
        subj_role = self._role_in_slot(nps, ACTIVE, "subject")  # kartā
        subj_plural = subj_role is not None and str(by_role[subj_role]["number"]) != "eka"
        verb_form = self._verb_form(verb, subj_plural)
        if verb_form is None:
            return None
        # Surface order: subject, verb, object, then obliques (SVO + PPs).
        chunks: list[tuple[str, str]] = []
        if subj_role is not None:
            chunks.append((nps[subj_role][0], subj_role))
        chunks.append((verb_form, "kriyA"))
        obj_role = self._role_in_slot(nps, ACTIVE, "object")  # karma
        if obj_role is not None:
            chunks.append((nps[obj_role][0], obj_role))
        for role in OBLIQUE_ORDER:
            if role in nps:
                phrase = self._oblique_part(role, ACTIVE, nps)
                if phrase is None:
                    return None
                chunks.append((phrase, role))
        return chunks

    def _passive_chunks(
        self, by_role: dict[str, Word], nps: dict[str, tuple[str, str]], verb: Word
    ) -> list[tuple[str, str]] | None:
        # The patient takes the subject slot under the passive; no patient → no
        # natural English passive.
        subj_role = self._role_in_slot(nps, PASSIVE, "subject")  # karma
        if subj_role is None:
            return None
        participle = PARTICIPLE.get(str(verb["dhatu"]))
        if participle is None:
            return None
        # The auxiliary agrees with the patient (the new surface subject).
        subj_plural = str(by_role[subj_role]["number"]) != "eka"
        aux = self._be(subj_plural, str(verb.get("lakara", "")))
        if aux is None:
            return None
        chunks: list[tuple[str, str]] = [
            (nps[subj_role][0], subj_role),
            (f"{aux} {participle}", "kriyA"),
        ]
        agent_role = self._role_in_slot(nps, PASSIVE, "agent")  # kartā → "by …"
        if agent_role is not None:
            phrase = self._oblique_part(agent_role, PASSIVE, nps)
            if phrase is None:
                return None
            chunks.append((phrase, agent_role))
        for role in OBLIQUE_ORDER:
            if role in nps:
                phrase = self._oblique_part(role, PASSIVE, nps)
                if phrase is None:
                    return None
                chunks.append((phrase, role))
        return chunks

    def stream(self, n: int, *, seed: int = 0) -> Iterator[AnnotatedSentence]:
        """Yield up to ``n`` gold-annotated English sentences (deterministic by seed)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return
        emitted = 0
        for frame in enumerate_frames(n * 2 + 64, seed=seed):
            if emitted >= n:
                break
            sentence = self.realize(frame)
            if sentence is None:
                continue
            yield sentence
            emitted += 1


def build_contrast_set(
    frame: KarakaFrame, realizer: EnglishFrameRealizer | None = None
) -> list[AnnotatedSentence]:
    """Realize one frame across constructions that share its gold role set.

    The H1_CONTRAST training unit: the same kāraka frame rendered with the surface
    varied but the roles held invariant — active voice plus, for transitive
    frames, the karmaṇi passive. Members with no valid realization are skipped, so
    an intransitive frame yields the active member alone.
    """
    r = realizer or EnglishFrameRealizer()
    members: list[AnnotatedSentence] = []
    for voice in ("active", "passive"):
        out = r.realize(frame, voice=voice)
        if out is not None:
            members.append(out)
    return members
