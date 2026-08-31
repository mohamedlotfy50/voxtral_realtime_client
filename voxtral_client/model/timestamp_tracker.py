"""Tracks token positions and groups them into timestamped words.

Uses the server-provided ``delta`` text for word content (no local tokenizer
needed). Token ids ``32`` (``[STREAMING_PAD]``) and ``33`` (``[STREAMING_WORD]``)
mark silence and word boundaries; ids ``>= 1000`` are word content.

If a local ``tekken.json`` is available (via ``model_path``) it is used as a
fallback for decoding when delta text is empty.
"""
from __future__ import annotations

import os
from typing import Any

from voxtral_client.constants import (
    DEFAULT_DELAY_TOKENS,
    SPECIAL_TOKEN_THRESHOLD,
    STREAMING_PAD_ID,
    STREAMING_WORD_ID,
)
from voxtral_client.model.word_timestamp import WordTimestamp


class TimestampTracker:
    def __init__(
        self,
        delay_tokens: int = DEFAULT_DELAY_TOKENS,
        model_path: str | None = None,
    ) -> None:
        self.delay_tokens = delay_tokens
        self.model_path = model_path
        self.token_position: int = 0
        self.current_word_tokens: list[int] = []
        self.current_word_delta: str = ""
        self.current_word_start: int = -1
        self.consecutive_pads: int = 0
        self.current_word_silence: int = 0
        self.words: list[WordTimestamp] = []
        self._tokenizer: Any = None
        self._tokenizer_loaded: bool = False

    def process_tokens(
        self, token_ids: list[int], delta_text: str = ""
    ) -> list[WordTimestamp]:
        """Process a batch of token ids with server-decoded delta text.

        Returns the words completed by this batch.
        """
        completed: list[WordTimestamp] = []
        delta_chars = list(delta_text)

        for tid in token_ids:
            if tid == STREAMING_PAD_ID:
                if self.current_word_start >= 0:
                    self._finish_word(completed)
                self.consecutive_pads += 1
            elif tid == STREAMING_WORD_ID:
                if self.current_word_start >= 0:
                    self._finish_word(completed)
                self.current_word_start = self.token_position
                self.current_word_tokens = []
                self.current_word_delta = ""
                self.current_word_silence = self.consecutive_pads
                self.consecutive_pads = 0
            elif tid >= SPECIAL_TOKEN_THRESHOLD:
                if self.current_word_start >= 0:
                    self.current_word_tokens.append(tid)
                    if delta_chars:
                        self.current_word_delta += delta_chars.pop(0)
            self.token_position += 1

        if delta_chars and self.current_word_start >= 0:
            self.current_word_delta += "".join(delta_chars)

        return completed

    def finalize(self) -> list[WordTimestamp]:
        """Flush any in-progress word at end of stream."""
        completed: list[WordTimestamp] = []
        if self.current_word_start >= 0:
            self._finish_word(completed)
        return completed

    def _finish_word(self, completed: list[WordTimestamp]) -> None:
        if self.current_word_start < 0:
            return

        word_text = self.current_word_delta.strip()
        if not word_text and self.current_word_tokens:
            word_text = self._decode_word_local(self.current_word_tokens)

        if word_text:
            w = WordTimestamp(
                text=word_text,
                start_token=self.current_word_start,
                end_token=self.token_position - 1,
                preceding_silence_tokens=self.current_word_silence,
            )
            self.words.append(w)
            completed.append(w)

        self.current_word_tokens = []
        self.current_word_delta = ""
        self.current_word_start = -1
        self.current_word_silence = 0

    def _decode_word_local(self, token_ids: list[int]) -> str:
        """Fallback: decode token ids locally using ``tekken.json``."""
        if not self._tokenizer_loaded:
            self._load_tokenizer()
        if self._tokenizer is None:
            return ""
        try:
            adjusted = [t - 1000 for t in token_ids]
            return self._tokenizer._model.decode(adjusted).strip()
        except Exception:
            return ""

    def _load_tokenizer(self) -> None:
        self._tokenizer_loaded = True
        if not self.model_path:
            return
        try:
            tekken_path = os.path.join(self.model_path, "tekken.json")
            if not os.path.exists(tekken_path):
                return
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

            self._tokenizer = MistralTokenizer.from_file(
                tekken_path
            ).instruct_tokenizer.tokenizer
            self._tokenizer.special_token_policy = SpecialTokenPolicy.IGNORE
        except Exception:
            pass
