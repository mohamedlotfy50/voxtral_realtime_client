"""Receives realtime transcription events and drives the display.

Owns the event dispatch loop, the timestamp tracker / segment collector, and
— when diarization is enabled — the per-segment speaker-labeling + ANSI redraw
flow. Pure rendering is delegated to :class:`TranscriptionDisplay`.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from voxtral_client.constants import STREAMING_PAD_ID, STREAMING_WORD_ID, TOKEN_MS
from voxtral_client.model.word_timestamp import WordTimestamp
from voxtral_client.presentation.display import TranscriptionDisplay

if TYPE_CHECKING:
    from voxtral_client.audio.mic_buffer import MicRingBuffer
    from voxtral_client.model.segment_collector import SegmentCollector
    from voxtral_client.model.timestamp_tracker import TimestampTracker
    from voxtral_client.net.diarization_client import DiarizationClient
    from voxtral_client.net.realtime_client import RealtimeClient


class TranscriptionReceiver:
    def __init__(
        self,
        client: "RealtimeClient",
        tracker: "TimestampTracker",
        segment_collector: "SegmentCollector | None",
        timestamps: bool,
        segment_mode: str,
        show_tokens: bool,
        display: TranscriptionDisplay,
        ring: "MicRingBuffer | None" = None,
        diar: "DiarizationClient | None" = None,
    ) -> None:
        self._client = client
        self._tracker = tracker
        self._segment_collector = segment_collector
        self._timestamps = timestamps
        self._segment_mode = segment_mode
        self._show_tokens = show_tokens
        self._display = display
        self._ring = ring
        self._diar = diar
        self._first_row = True
        self._rows_printed = 0

    async def run(self) -> None:
        while True:
            response = await self._client.recv_json()
            if response is None:
                continue
            rtype = response.get("type")
            if rtype == "transcription.delta":
                await self._handle_delta(response)
            elif rtype == "transcription.done":
                await self._handle_done(response)
                return
            elif rtype == "error":
                print(f"\n[error] {response.get('error', response)}")
                return
            elif rtype in ("session.created", "session.updated"):
                print(f"[{rtype}] id={response.get('id', '?')}")

    async def _handle_delta(self, response: dict) -> None:
        token_ids = response.get("token_ids", [])
        delta_text = response.get("delta", "")

        if self._show_tokens and token_ids:
            self._print_tokens(token_ids, delta_text)

        if not self._timestamps:
            print(delta_text, end="", flush=True)
            return

        completed_words = self._tracker.process_tokens(token_ids, delta_text)

        if self._segment_mode == "word":
            for w in completed_words:
                self._print_word(w)
        elif self._segment_mode == "segment" and self._segment_collector is not None:
            for w in completed_words:
                for seg in self._segment_collector.add_word(w):
                    self._first_row = await self._emit_segment(seg)

    async def _handle_done(self, response: dict) -> None:
        if not self._timestamps:
            print()
            if response.get("usage"):
                print(f"[usage] {response['usage']}")
            print("[done]")
            return

        final_words = self._tracker.finalize()

        if self._segment_mode == "word":
            for w in final_words:
                self._print_word(w)
        elif self._segment_mode == "segment" and self._segment_collector is not None:
            for w in final_words:
                for seg in self._segment_collector.add_word(w):
                    self._first_row = await self._emit_segment(seg)
            for seg in self._segment_collector.finalize():
                self._first_row = await self._emit_segment(seg)

        self._print_summary(response)

        if self._segment_mode == "segment" and self._segment_collector is not None:
            await self._print_final_segments()

    async def _emit_segment(self, seg) -> bool:
        """Label + print one segment row; returns the updated first_row flag."""
        if self._first_row:
            self._display.print_segment_header()
            self._first_row = False

        if self._diar is not None and self._ring is not None:
            pcm = self._ring.slice(seg.start_s, seg.end_s)
            if pcm is not None and pcm.shape[0] > 0:
                try:
                    res = await self._diar.label(pcm, seg.start_s, seg.end_s)
                    seg.speaker = res["speaker"]
                    relabel = res.get("relabel")
                    if relabel and self._segment_collector is not None:
                        mapping = relabel.get("mapping", {})
                        self._display.apply_relabel(
                            self._segment_collector.segments, mapping
                        )
                        self._display.apply_segment_patch(
                            self._segment_collector.segments,
                            relabel.get("segments") or [],
                        )
                        self._display.redraw_rows(
                            self._segment_collector.segments, self._rows_printed
                        )
                except Exception as e:
                    print(f"  [diarize error] {e}", file=sys.stderr)

        print(self._display.segment_row(seg))
        self._rows_printed += 1
        return self._first_row

    def _print_word(self, word: WordTimestamp) -> None:
        if self._first_row:
            self._display.print_word_header()
            self._first_row = False
        print(self._display.word_row(word))

    def _print_tokens(self, token_ids: list[int], delta_text: str) -> None:
        labels = []
        for tid in token_ids:
            if tid == STREAMING_PAD_ID:
                labels.append("[PAD]")
            elif tid == STREAMING_WORD_ID:
                labels.append("[WORD]")
            else:
                labels.append(f"{tid}")
        print(f"  tokens: {labels}  -> {delta_text!r}")

    def _print_summary(self, response: dict) -> None:
        print(f"\n{'=' * 64}")
        if self._segment_mode == "segment" and self._segment_collector is not None:
            print(f"Total segments: {len(self._segment_collector.segments)}")
        print(f"Total words: {len(self._tracker.words)}")
        print(
            f"Total tokens: {self._tracker.token_position} "
            f"({self._tracker.token_position * TOKEN_MS / 1000:.1f}s of audio)"
        )
        print(
            f"Delay: {self._tracker.delay_tokens} tokens "
            f"({self._tracker.delay_tokens * TOKEN_MS:.0f}ms)"
        )
        if response.get("usage"):
            print(f"Usage: {response['usage']}")

    async def _print_final_segments(self) -> None:
        assert self._segment_collector is not None
        if self._diar is not None:
            try:
                cons = await self._diar.consolidate()
                relabel = cons.get("relabel")
                if relabel:
                    self._display.apply_relabel(
                        self._segment_collector.segments, relabel.get("mapping", {})
                    )
                    self._display.apply_segment_patch(
                        self._segment_collector.segments,
                        relabel.get("segments") or [],
                    )
                    self._display.redraw_rows(
                        self._segment_collector.segments, self._rows_printed
                    )
            except Exception as e:
                print(f"  [consolidate error] {e}", file=sys.stderr)
            print("\n--- Speaker summary ---")
            try:
                summ = await self._diar.summary()
                for sp in summ.get("speakers", []):
                    print(
                        f"  {sp['id']}: {sp['talk_time_s']}s "
                        f"({sp['utterances']} utterances)"
                    )
            except Exception as e:
                print(f"  [summary error] {e}")
        print("\nFull transcription:")
        for s in self._segment_collector.segments:
            label = f"{s.speaker}: " if s.speaker else ""
            print(f"  {label}{s.text}")
