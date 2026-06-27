"""Kāraka ↔ vibhakti ↔ English construction index (H1_CONTRAST, ADR-0043).

The vibhakti (case) system is the complete enumeration of how each kāraka
(semantic role) surfaces. This module is the single source of truth for two
mappings, kept in one place so the Sanskrit and English realizers cannot drift:

1. **kāraka → vibhakti** — which Sanskrit case a role takes in a given voice. The
   load-bearing cells are the voice shift: under the karmaṇi passive the agent
   moves nominative → instrumental and the patient moves accusative → nominative.
   This decoupling of deep role from surface case is the mechanism's whole point.
2. **kāraka → English device** — how the role surfaces in English, which has
   almost no case morphology and so leans on word order (subject/object) and
   prepositions (with/to/from/in) plus the passive by-phrase.

Because English collapses distinctions Sanskrit keeps (instrument, comitative,
and means all surface as "with"), :func:`english_collapses` audits a construction
set for realizations that English cannot tell apart, so they can be dropped and
logged rather than silently mislabeled.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

_LOGGER = logging.getLogger(__name__)

ACTIVE = "active"
PASSIVE = "passive"

#: The six kārakas, in canonical order (WX role names).
ALL_KARAKAS: tuple[str, ...] = (
    "karwA",  # kartā — agent
    "karma",  # karma — patient
    "karaNam",  # karaṇa — instrument
    "sampraxAnam",  # sampradāna — recipient
    "apAxAnam",  # apādāna — source
    "aXikaraNam",  # adhikaraṇa — locus
)


@dataclass(frozen=True)
class Construction:
    """How one kāraka surfaces in one voice: Sanskrit case + English device."""

    karaka: str  # WX role name
    voice: str  # ACTIVE | PASSIVE
    vibhakti: str  # Vidyut Vibhakti enum name (Prathama, Dvitiya, …)
    slot: str  # "subject" | "object" | "agent" | "oblique"
    preposition: str  # English preposition; "" when positional


#: Core-argument cells depend on voice — the kāraka→vibhakti shift that *is* the
#: contrast: (kāraka, voice) → (vibhakti, English slot, English preposition).
_CORE: dict[tuple[str, str], tuple[str, str, str]] = {
    ("karwA", ACTIVE): ("Prathama", "subject", ""),
    ("karma", ACTIVE): ("Dvitiya", "object", ""),
    ("karwA", PASSIVE): ("Trtiya", "agent", "by "),
    ("karma", PASSIVE): ("Prathama", "subject", ""),
}

#: Oblique kārakas: case is fixed by the role, independent of voice.
_OBLIQUE: dict[str, tuple[str, str, str]] = {
    "karaNam": ("Trtiya", "oblique", "with "),
    "sampraxAnam": ("Caturthi", "oblique", "to "),
    "apAxAnam": ("Panchami", "oblique", "from "),
    "aXikaraNam": ("Saptami", "oblique", "in "),
}


def construction_for(karaka: str, voice: str) -> Construction | None:
    """Return the construction for a kāraka in a voice, or ``None`` if unmapped."""
    core = _CORE.get((karaka, voice))
    if core is not None:
        vibhakti, slot, preposition = core
        return Construction(karaka, voice, vibhakti, slot, preposition)
    oblique = _OBLIQUE.get(karaka)
    if oblique is not None:
        vibhakti, slot, preposition = oblique
        return Construction(karaka, voice, vibhakti, slot, preposition)
    return None


def english_collapses(constructions: Iterable[Construction]) -> list[list[Construction]]:
    """Group constructions that realize identically in English.

    Two constructions collapse when they share the same (slot, preposition):
    English offers no surface cue to tell them apart, so labeling either is
    unsafe. Returns one group per colliding realization (each with ≥2 members);
    an empty list means every construction is distinguishable. Within the
    six-kāraka inventory at full-NP granularity, no collapse occurs — the audit
    exists to catch collapses introduced by finer roles or pronouns.
    """
    by_realization: dict[tuple[str, str], list[Construction]] = {}
    for c in constructions:
        by_realization.setdefault((c.slot, c.preposition), []).append(c)
    return [group for group in by_realization.values() if len(group) > 1]


def realized_constructions(voice: str) -> list[Construction]:
    """Every kāraka's construction in ``voice`` (the constructions actually used)."""
    out: list[Construction] = []
    for karaka in ALL_KARAKAS:
        construction = construction_for(karaka, voice)
        if construction is not None:
            out.append(construction)
    return out


def collapse_warning(group: list[Construction]) -> str:
    """A drop message for one colliding realization: keep the first, drop the rest."""
    kept = group[0]
    members = ", ".join(c.karaka for c in group)
    return (
        f"english collapse [{kept.voice}]: {{{members}}} all surface as "
        f"'{kept.preposition}{kept.slot}'; keeping {kept.karaka}, dropping the rest"
    )


def log_coverage(logger: logging.Logger | None = None) -> None:
    """Log construction coverage per voice, warning on any English collapse.

    Satisfies the "coverage is reported, never silently truncated" requirement: a
    generation run calls this so the kāraka count and any dropped (collapsing)
    constructions are visible in the run record. For the six-kāraka full-NP
    inventory no collapse occurs, so this reports full coverage and zero drops.
    """
    log = logger or _LOGGER
    for voice in (ACTIVE, PASSIVE):
        cons = realized_constructions(voice)
        groups = english_collapses(cons)
        for group in groups:
            log.warning("%s", collapse_warning(group))
        log.info(
            "construction coverage [%s]: %d kārakas realized, %d collapse-group(s)",
            voice,
            len(cons),
            len(groups),
        )
