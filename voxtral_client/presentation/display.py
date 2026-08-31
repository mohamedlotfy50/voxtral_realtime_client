"""Pure rendering for transcription output (no I/O scheduling, no network).

Handles the three display modes — plain inline text, per-word timestamp table,
per-segment timestamp table — plus ANSI cursor redraw used when the
diarization server relabels already-printed segments.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from voxtral_client.model.segment_timestamp import SegmentTimestamp
from voxtral_client.model.word_timestamp import WordTimestamp

if TYPE_CHECKING:
    from voxtral_client.nlp.language_detection import LanguageDetector


class TranscriptionDisplay:
    def __init__(
        self,
        detect_language: bool,
        lang_det: "LanguageDetector | None",
    ) -> None:
        self.detect_language = detect_language
        self.lang_det = lang_det

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m:02d}:{s:06.3f}"

    def print_word_header(self) -> None:
        print(f"\n{'Word':<30} {'Start':>12} {'End':>12}")
        print(f"{'-' * 30} {'-' * 12} {'-' * 12}")

    def print_segment_header(self) -> None:
        print(f"\n{'Speaker':<14}{'Segment':<44} {'Start':>12} {'End':>12}")
        print(f"{'-' * 14}{'-' * 44} {'-' * 12} {'-' * 12}")

    def word_row(self, word: WordTimestamp) -> str:
        lang_tag = self._lang_tag(word.text)
        return (
            f"{lang_tag}{word.text:<30} "
            f"{self.format_timestamp(word.start_s):>12} "
            f"{self.format_timestamp(word.end_s):>12}"
        )

    def segment_row(self, seg: SegmentTimestamp) -> str:
        speaker = f"{seg.speaker} " if seg.speaker else ""
        lang_tag = self._lang_tag(seg.text)
        return (
            f"{speaker:<14}{lang_tag}{seg.text:<44.44} "
            f"{self.format_timestamp(seg.start_s):>12} "
            f"{self.format_timestamp(seg.end_s):>12}"
        )

    def redraw_rows(self, segments: list[SegmentTimestamp], n_rows: int) -> None:
        """ANSI-redraw the last ``n_rows`` printed segment rows in place."""
        if n_rows <= 0:
            return
        out = sys.stdout
        out.write(f"\033[{n_rows}A")
        for i in range(n_rows):
            out.write("\033[2K")
            out.write(self.segment_row(segments[i]) + "\n")
        out.flush()

    @staticmethod
    def apply_relabel(segments: list[SegmentTimestamp], mapping: dict) -> int:
        changed = 0
        for s in segments:
            if s.speaker and s.speaker in mapping and mapping[s.speaker] != s.speaker:
                s.speaker = mapping[s.speaker]
                changed += 1
        return changed

    @staticmethod
    def apply_segment_patch(segments: list[SegmentTimestamp],
                            patch: list[dict]) -> int:
        """Apply per-segment relabel corrections (newly confirmed segments).

        The bulk ``mapping`` cannot express these: the same borrowed
        temporary label may belong to different real speakers once a
        pipeline pass confirms them. Patch entries match segments by
        ``(start_s, end_s)``.
        """
        by_time = {(s.start_s, s.end_s): s for s in segments}
        changed = 0
        for p in patch:
            s = by_time.get((p.get("start_s"), p.get("end_s")))
            if s is not None and p.get("speaker") and s.speaker != p["speaker"]:
                s.speaker = p["speaker"]
                changed += 1
        return changed

    def _lang_tag(self, text: str) -> str:
        if not self.detect_language or self.lang_det is None:
            return ""
        return f"[{self.lang_det.detect(text).upper()}] "
