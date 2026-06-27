"""Vidyut-native frame → sentence realizer (ADR-0033).

This is the *realization* port that supersedes the un-provisionable Saṃsādhanī
HTTP generator. Given a :class:`~psalm.domain.data.karaka_frames.KarakaFrame`
(a meaning structure: nominals tagged with kāraka roles + a verb with voice and
tense), it produces a grammatically inflected Sanskrit sentence **and** the gold
kāraka parse, by driving the ``vidyut.prakriya`` Pāṇinian derivation engine.

Two properties matter for PSALM and are enforced here:

1. **Gold kāraka for free.** The realizer *builds* the sentence from the frame,
   so the (surface-pada, role) alignment is gold by construction — not recovered
   by a parser. ``AnnotatedSentence.karaka_parse`` is therefore lossless w.r.t.
   the frame's role assignment.
2. **No fabrication.** Every surface form is a real Vidyut derivation. Forward
   sandhi applies only well-defined ach-sandhi (vowel coalescence) rules; any
   junction we cannot resolve by a rule is left as a word boundary and the
   sentence is flagged ``meta["sandhi"]="partial"`` (padapāṭha). We never invent
   a fused form. Grammaticality is the derivation engine's, not a mock's.

Encoding: surfaces are emitted in **SLP1** (ASCII, matches the native Vidyut
channel and the existing :mod:`vidyut_source` adapter). ``meta["scheme"]`` records
this so downstream consumers can transliterate.

The lexicon below is *verified*: every (stem, liṅga, vibhakti) and every
(dhātu, lakāra) entry has been checked to derive a correct classical form (see
``tests/infrastructure/test_vidyut_realizer.py``). Frame strings are a WX/SLP1
hybrid (consonants in WX: ``w``→t, ``x``→d; vowels already SLP1: ``f``→ṛ), so we
map them explicitly rather than guessing a transliteration scheme.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import cast

from psalm.application.data.ports import AnnotatedSentence
from psalm.domain.data.karaka_frames import AKARMAKA_DHATUS, KarakaFrame, enumerate_frames
from psalm.domain.linguistics.construction_index import construction_for

Word = dict[str, object]


def _frame_words(frame: KarakaFrame) -> list[Word]:
    return cast("list[Word]", frame.structure.get("words", []))


class VidyutUnavailableError(RuntimeError):
    """Raised when the optional ``vidyut`` dependency is not installed."""


@dataclass(frozen=True)
class StemEntry:
    """A verified nominal stem: SLP1 prātipadika + liṅga + nyāp requirement."""

    slp1: str
    linga: str  # "pum" | "napum" | "stri"
    nyap: bool  # feminine ā/ī stems need Pratipadika.nyap to decline correctly


@dataclass(frozen=True)
class DhatuEntry:
    """A verified dhātu: SLP1 upadeśa + gaṇa name."""

    upadesha: str
    gana: str  # vidyut.prakriya.Gana attribute name


#: WX/SLP1 frame stem → verified SLP1 prātipadika + liṅga + nyāp flag.
STEMS: dict[str, StemEntry] = {
    "rAma": StemEntry("rAma", "pum", False),
    "nara": StemEntry("nara", "pum", False),
    "bAla": StemEntry("bAla", "pum", False),
    "guru": StemEntry("guru", "pum", False),
    "aSva": StemEntry("aSva", "pum", False),
    "vana": StemEntry("vana", "napum", False),
    "Pala": StemEntry("Pala", "napum", False),
    "jala": StemEntry("jala", "napum", False),
    "puswaka": StemEntry("pustaka", "napum", False),
    "gfha": StemEntry("gfha", "napum", False),
    "siwA": StemEntry("sitA", "stri", True),
    "kanyA": StemEntry("kanyA", "stri", True),
    "naxI": StemEntry("nadI", "stri", True),
    "vixyA": StemEntry("vidyA", "stri", True),
}

#: WX dhātu code (root + gaṇa digit) → verified SLP1 upadeśa + gaṇa.
DHATUS: dict[str, DhatuEntry] = {
    "gam1": DhatuEntry("ga\\mx~", "Bhvadi"),
    "paW1": DhatuEntry("paWa~", "Bhvadi"),
    "KAx1": DhatuEntry("KAdf~", "Bhvadi"),
    "xfS1": DhatuEntry("df\\Si~r", "Bhvadi"),
    "kf8": DhatuEntry("qukf\\Y", "Tanadi"),
    "BU1": DhatuEntry("BU", "Bhvadi"),
    "sWA1": DhatuEntry("zWA\\", "Bhvadi"),
    "vax1": DhatuEntry("vada~", "Bhvadi"),
    "xA1": DhatuEntry("qudA\\Y", "Juhotyadi"),
    "vas1": DhatuEntry("vasa~", "Bhvadi"),
}

#: kāraka role → vibhakti (case) name. The kāraka→vibhakti map is the heart of
#: the Pāṇinian realization: prathamā for kartā (kartari), dvitīyā for karma, etc.
VIBHAKTI_FOR_KARAKA: dict[str, str] = {
    "karwA": "Prathama",
    "karma": "Dvitiya",
    "karaNam": "Trtiya",
    "sampraxAnam": "Caturthi",
    "apAxAnam": "Panchami",
    "sambanXa": "Sasthi",
    "aXikaraNam": "Saptami",
    "predicate": "Prathama",
}


def _vibhakti_for(karaka: str, voice: str) -> str | None:
    """Vibhakti for a kāraka in a voice.

    The six kārakas are sourced from the shared construction index (the single
    source of truth, so the Sanskrit and English realizers cannot drift); under
    the karmaṇi passive this shifts kartā → instrumental and karma → nominative.
    ``predicate`` and ``sambandha`` fall back to the local active-voice map.
    """
    construction = construction_for(karaka, voice)
    if construction is not None:
        return construction.vibhakti
    return VIBHAKTI_FOR_KARAKA.get(karaka)


#: frame number → vacana name.
VACANA: dict[str, str] = {"eka": "Eka", "xvi": "Dvi", "bahu": "Bahu"}

#: frame gender → liṅga name.
LINGA: dict[str, str] = {"pum": "Pum", "napum": "Napumsaka", "stri": "Stri"}

#: WX lakāra label → vidyut.prakriya.Lakara attribute name.
LAKARA: dict[str, str] = {
    "varwamAnaH": "Lat",  # present
    "anaxyawanaBUwaH": "Lan",  # imperfect
    "sAmAnyaBaviRyakAlaH": "Lrt",  # future
    "viXiH": "VidhiLin",  # optative
    "AjFA": "Lot",  # imperative
}

#: Strictly intransitive (akarmaka) dhātus that must NEVER license a karma —
#: re-exported from the domain lexicon (single source of truth, ADR-0033). Used
#: by :func:`frame_transitivity_violation` and the realizer's guard.
AKARMAKA: frozenset[str] = AKARMAKA_DHATUS

#: SLP1 simple + diphthong vowels.
_VOWELS = set("aAiIuUfFxXeEoO")
#: SLP1 short → savarṇa class representative, for savarṇa-dīrgha.
_A = set("aA")
_I = set("iI")
_U = set("uU")
_R = set("fF")


def _ach_sandhi(left: str, right: str) -> str | None:
    """Return the ach-sandhi (vowel-coalescence) fusion of two padas, or ``None``.

    Implements only the well-defined, unambiguous saṃhitā vowel rules at a word
    junction where ``left`` ends in a vowel and ``right`` begins with a vowel:
    savarṇa-dīrgha, guṇa, vṛddhi, and yaṇ. Returns ``None`` (caller keeps a word
    boundary) when either side is not vowel-adjacent or the rule would be
    ambiguous (e.g. e/o + vowel → avagraha), so we never fabricate a fusion.
    """
    if not left or not right:
        return None
    a, b = left[-1], right[0]
    if a not in _VOWELS or b not in _VOWELS:
        return None
    stem_l, stem_r = left[:-1], right[1:]

    def join(mid: str) -> str:
        return f"{stem_l}{mid}{stem_r}"

    # savarṇa-dīrgha: like + like → long
    if a in _A and b in _A:
        return join("A")
    if a in _I and b in _I:
        return join("I")
    if a in _U and b in _U:
        return join("U")
    if a in _R and b in _R:
        return join("F")
    # guṇa / vṛddhi: a/ā + heterogeneous vowel
    if a in _A:
        if b in _I:
            return join("e")
        if b in _U:
            return join("o")
        if b in _R:
            return join("ar")
        if b == "x":
            return join("al")
        if b in ("e", "E"):
            return join("E")
        if b in ("o", "O"):
            return join("O")
        return None
    # yaṇ: i/u/ṛ + heterogeneous vowel → y/v/r + vowel
    if a in _I:
        return f"{stem_l}y{b}{stem_r}"
    if a in _U:
        return f"{stem_l}v{b}{stem_r}"
    if a in _R:
        return f"{stem_l}r{b}{stem_r}"
    # e/E/o/O + vowel → avagraha / hiatus: ambiguous, leave a boundary.
    return None


def forward_sandhi(padas: Sequence[str]) -> tuple[str, str]:
    """Join padas with forward ach-sandhi; report the sandhi mode.

    Returns ``(text, mode)`` where ``mode`` is ``"full"`` (every junction fused
    by a rule), ``"partial"`` (at least one junction left as a space — padapāṭha),
    or ``"none"`` (single pada / nothing to join). No fusion is ever fabricated:
    unresolved junctions become spaces.
    """
    padas = [p for p in padas if p]
    if not padas:
        return "", "none"
    if len(padas) == 1:
        return padas[0], "none"
    out = padas[0]
    any_space = False
    for nxt in padas[1:]:
        fused = _ach_sandhi(out, nxt)
        if fused is None:
            out = f"{out} {nxt}"
            any_space = True
        else:
            out = fused
    return out, ("partial" if any_space else "full")


def frame_transitivity_violation(frame: KarakaFrame) -> bool:
    """True iff the frame pairs a non-transitive dhātu with a karma noun.

    Native (expanded-lexicon) frames carry a DCS-grounded ``transitive`` flag on
    the verb; legacy frames fall back to the WX akarmaka set.
    """
    words = _frame_words(frame)
    has_karma = any(w.get("pos") == "noun" and w.get("karaka") == "karma" for w in words)
    if not has_karma:
        return False
    for w in words:
        if w.get("pos") != "verb":
            continue
        if "transitive" in w:
            if not bool(w["transitive"]):
                return True
        elif w.get("dhatu") in AKARMAKA:
            return True
    return False


@dataclass(frozen=True)
class VidyutRealizerConfig:
    """Realizer configuration."""

    #: kāraka ordering for surface linearization; verb appended last (SOV-ish).
    word_order: tuple[str, ...] = (
        "karwA",
        "apAxAnam",
        "sampraxAnam",
        "karaNam",
        "aXikaraNam",
        "karma",
    )
    apply_sandhi: bool = True
    include_derivation: bool = True


class VidyutFrameRealizer:
    """Realize :class:`KarakaFrame`s as Sanskrit sentences via ``vidyut.prakriya``.

    Implements the ``SentenceGenerator`` port: :meth:`stream` enumerates frames
    (deterministic by ``seed``) and realizes each, yielding gold-annotated
    sentences. :meth:`realize` exposes single-frame realization for tests.
    """

    def __init__(self, config: VidyutRealizerConfig | None = None) -> None:
        self.config = config or VidyutRealizerConfig()
        self._vyakarana: object = None
        self._mod: object = None

    def _vidyut(self) -> object:
        if self._mod is None:
            try:
                import vidyut.prakriya as prakriya
            except ImportError as exc:  # pragma: no cover - only without vidyut
                raise VidyutUnavailableError(
                    "vidyut is not installed. Install it: `uv pip install vidyut`."
                ) from exc
            self._mod = prakriya
            self._vyakarana = prakriya.Vyakarana()
        return self._mod

    def _decline(self, noun: Word, voice: str = "active") -> str | None:
        p = self._vidyut()
        # Native (expanded-lexicon) frames carry SLP1 + liṅga + nyāp directly and
        # bypass the small WX-keyed STEMS table; legacy frames look up STEMS.
        entry: StemEntry | None
        if "slp1" in noun:
            entry = StemEntry(
                slp1=str(noun["slp1"]),
                linga=str(noun["linga"]),
                nyap=bool(noun.get("nyap", False)),
            )
        else:
            entry = STEMS.get(str(noun["stem"]))
            if entry is None:
                return None
        karaka = str(noun["karaka"])
        vibhakti_name = _vibhakti_for(karaka, voice)
        if vibhakti_name is None:
            return None
        vacana_name = VACANA[str(noun["number"])]
        linga_name = LINGA[entry.linga]
        pratipadika = (
            p.Pratipadika.nyap(entry.slp1) if entry.nyap else p.Pratipadika.basic(entry.slp1)
        )
        args = p.Pada.Subanta(
            pratipadika=pratipadika,
            linga=getattr(p.Linga, linga_name),
            vibhakti=getattr(p.Vibhakti, vibhakti_name),
            vacana=getattr(p.Vacana, vacana_name),
        )
        results = self._vyakarana.derive(args)
        if not results:
            return None
        # Deterministic: first derivation. Multiple results are sandhi variants of
        # one correct form (e.g. narAt/narAd); the first is canonical.
        return str(results[0].text)

    def _conjugate(
        self, verb: Word, vacana_label: str, voice: str = "active"
    ) -> tuple[str | None, tuple[str, ...]]:
        p = self._vidyut()
        # Native frames carry the upadeśa + gaṇa directly; legacy frames look up DHATUS.
        if "aupadeshika" in verb:
            entry: DhatuEntry | None = DhatuEntry(
                upadesha=str(verb["aupadeshika"]), gana=str(verb["gana"])
            )
        else:
            entry = DHATUS.get(str(verb["dhatu"]))
        if entry is None:
            return None, ()
        lakara_name = LAKARA.get(str(verb["lakara"]))
        if lakara_name is None:
            return None, ()
        dhatu = p.Dhatu.mula(entry.upadesha, getattr(p.Gana, entry.gana))
        prayoga = p.Prayoga.Karmani if voice == "passive" else p.Prayoga.Kartari
        args = p.Pada.Tinanta(
            dhatu=dhatu,
            prayoga=prayoga,
            lakara=getattr(p.Lakara, lakara_name),
            purusha=p.Purusha.Prathama,  # nominal subjects are 3rd person
            vacana=getattr(p.Vacana, VACANA[vacana_label]),
        )
        results = self._vyakarana.derive(args)
        if not results:
            return None, ()
        prakriya = results[0]
        text = str(prakriya.text)
        if not text:
            return None, ()
        derivation: tuple[str, ...] = ()
        if self.config.include_derivation:
            derivation = tuple(str(step.code) for step in prakriya.history)
        return text, derivation

    def realize(self, frame: KarakaFrame, voice: str = "active") -> AnnotatedSentence | None:
        """Realize one frame in ``voice`` (active | passive), or ``None``.

        ``None`` is returned (never a fabricated string) when: the frame violates
        transitivity (akarmaka + karma), a stem/dhātu is outside the verified
        lexicon, Vidyut yields no derivation, or a karmaṇi passive is requested of
        a frame with no karma. The gold kāraka parse is voice-invariant — only the
        surface (the kartā's case, the verb's prayoga) changes.
        """
        if frame_transitivity_violation(frame):
            return None
        words = _frame_words(frame)
        nouns = [w for w in words if w.get("pos") == "noun"]
        verbs = [w for w in words if w.get("pos") == "verb"]
        if len(verbs) != 1:
            return None
        verb = verbs[0]
        # The verb agrees with the kartā in the active voice and with the karma
        # (the new nominative subject) under the karmaṇi passive.
        if voice == "passive":
            karma = next((nw for nw in nouns if nw.get("karaka") == "karma"), None)
            if karma is None:
                return None
            agreement_vacana = str(karma["number"])
        else:
            karta = next((nw for nw in nouns if nw.get("karaka") == "karwA"), None)
            agreement_vacana = str(karta["number"]) if karta else "eka"

        realized: list[tuple[str, str, int]] = []
        for nw in nouns:
            surface = self._decline(nw, voice)
            if surface is None:
                return None
            order = self.config.word_order
            karaka = str(nw["karaka"])
            rank = order.index(karaka) if karaka in order else len(order)
            realized.append((surface, karaka, rank))

        verb_surface, derivation = self._conjugate(verb, agreement_vacana, voice)
        if verb_surface is None:
            return None

        realized.sort(key=lambda t: t[2])
        noun_tokens = [s for s, _, _ in realized]
        tokens = [*noun_tokens, verb_surface]
        roles: tuple[tuple[str, str], ...] = (
            *[(s, r) for s, r, _ in realized],
            (verb_surface, "kriyA"),
        )

        if self.config.apply_sandhi:
            text, sandhi_mode = forward_sandhi(tokens)
        else:
            text, sandhi_mode = " ".join(tokens), "none"

        return AnnotatedSentence(
            text=text,
            language="sa",
            karaka_parse=roles,
            derivation=derivation,
            meta={
                "dhatu": str(verb["dhatu"]),
                "lakara": str(verb["lakara"]),
                "prayoga": "karmani" if voice == "passive" else str(verb.get("prayoga", "karwari")),
                "scheme": "slp1",
                "sandhi": sandhi_mode,
                "generator": "vidyut-realizer",
            },
        )

    def stream(self, n: int, *, seed: int = 0) -> Iterator[AnnotatedSentence]:
        """Yield up to ``n`` gold-annotated sentences (deterministic by seed)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return
        emitted = 0
        # Over-enumerate to absorb any underivable frames; in practice the
        # verified lexicon realizes ~100% of enumerated frames.
        for frame in enumerate_frames(n * 2 + 64, seed=seed):
            if emitted >= n:
                break
            sentence = self.realize(frame)
            if sentence is None:
                continue
            yield sentence
            emitted += 1
