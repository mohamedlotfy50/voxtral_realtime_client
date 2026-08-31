"""Groups :class:`WordTimestamp` objects into sentence-level segments.

Sentence boundaries are detected by checking whether a word ends with a
language-specific terminator (``. ! ? ؟ । ॥`` ...). When ``language`` is
``None`` (auto-detect) every terminator from every supported language is
considered so segmentation works regardless of the spoken language.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from voxtral_client.model.segment_timestamp import SegmentTimestamp
from voxtral_client.model.word_timestamp import WordTimestamp
from voxtral_client.nlp.sentence_terminator import SentenceTerminators


@dataclass
class SegmentCollector:
    language: str | None = None
    pause_threshold_tokens: int = 12
    terminators: set[str] = field(default_factory=lambda: SentenceTerminators.default())
    buffer: list[WordTimestamp] = field(default_factory=list)
    segments: list[SegmentTimestamp] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.language is None:
            self.terminators = SentenceTerminators.all()
        else:
            self.terminators = SentenceTerminators.for_language(self.language)

    def add_word(self, word: WordTimestamp) -> list[SegmentTimestamp]:
        """Add a word; return any segments completed by it (may be empty)."""
        completed: list[SegmentTimestamp] = []
        if (self.buffer
                and word.preceding_silence_tokens >= self.pause_threshold_tokens):
            completed.extend(self._flush())
        self.buffer.append(word)
        text = word.text.rstrip()
        if text and text[-1] in self.terminators:
            completed.extend(self._flush())
        return completed

    def finalize(self) -> list[SegmentTimestamp]:
        """Flush the remaining buffer at end of stream."""
        return self._flush()

    def _flush(self) -> list[SegmentTimestamp]:
        if not self.buffer:
            return []
        text = " ".join(w.text for w in self.buffer)
        seg = SegmentTimestamp(
            text=text,
            start_token=self.buffer[0].start_token,
            end_token=self.buffer[-1].end_token,
        )
        self.segments.append(seg)
        self.buffer = []
        return [seg]
