"""Sentence terminators per language and helpers for selecting a set.

Per language (ISO-639-1):
  - Arabic uses ``؟`` (U+061F).
  - Hindi uses ``।`` (U+0964 danda) and ``॥`` (U+0965 double danda).
When the language is unknown (auto-detect) the union of all terminators is
used so segmentation works regardless of which language is spoken.
"""
from __future__ import annotations


_SENTENCE_TERMINATORS: dict[str, set[str]] = {
    "en": {".", "!", "?"},
    "nl": {".", "!", "?"},
    "fr": {".", "!", "?"},
    "de": {".", "!", "?"},
    "it": {".", "!", "?"},
    "pt": {".", "!", "?"},
    "es": {".", "!", "?"},
    "ar": {".", "!", "؟"},
    "hi": {".", "!", "?", "।", "॥"},
}

_DEFAULT_TERMINATORS: set[str] = {".", "!", "?"}

_ALL_TERMINATORS: set[str] = set()
for _terms in _SENTENCE_TERMINATORS.values():
    _ALL_TERMINATORS |= _terms


class SentenceTerminators:
    """Accessor for language-specific sentence terminator character sets."""

    @staticmethod
    def default() -> set[str]:
        return set(_DEFAULT_TERMINATORS)

    @staticmethod
    def all() -> set[str]:
        return set(_ALL_TERMINATORS)

    @staticmethod
    def for_language(language: str) -> set[str]:
        return set(_SENTENCE_TERMINATORS.get(language, _DEFAULT_TERMINATORS))
