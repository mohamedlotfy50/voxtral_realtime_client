"""Language detection for transcribed text (client-side display only).

Uses ``lingua-language-detector``. Imported hard per project policy: the
``--detect-language`` feature fails fast if ``lingua`` is not installed.
The model itself auto-detects language from audio; this only pretty-prints
a per-segment tag.
"""
from __future__ import annotations

from lingua import Language, LanguageDetectorBuilder


class LanguageDetector:
    """Detects which of the 9 Voxtral languages a text segment is in."""

    _SUPPORTED = (
        Language.ENGLISH,
        Language.FRENCH,
        Language.SPANISH,
        Language.GERMAN,
        Language.ITALIAN,
        Language.PORTUGUESE,
        Language.DUTCH,
        Language.ARABIC,
        Language.HINDI,
    )

    def __init__(self) -> None:
        self._detector = LanguageDetectorBuilder.from_languages(
            *self._SUPPORTED
        ).build()

    def detect(self, text: str) -> str:
        """Return an ISO-639-1 code (``en``, ``fr`` ...) or ``?`` if unknown."""
        if not text.strip():
            return "?"
        result = self._detector.detect_language_of(text)
        if result is None:
            return "?"
        return result.iso_code_639_1.name.lower()
