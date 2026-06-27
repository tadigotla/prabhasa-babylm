"""Linguistic engines for kāraka role assignment.

The spaCy-based analyzer in :mod:`english_karaka_real` is re-exported **lazily**
(PEP 562): the names below resolve on first access, so pure, dependency-free
modules in this package (e.g. :mod:`construction_index`) can be imported without
pulling in spaCy. This mirrors the optional-dependency handling used for Vidyut.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .english_karaka_real import (
        TokenRole,
        assign_karaka_roles_spacy,
        parse_and_assign,
        roles_to_dict,
    )

__all__ = [
    "TokenRole",
    "assign_karaka_roles_spacy",
    "parse_and_assign",
    "roles_to_dict",
]

_LAZY = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import english_karaka_real

        return getattr(english_karaka_real, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
