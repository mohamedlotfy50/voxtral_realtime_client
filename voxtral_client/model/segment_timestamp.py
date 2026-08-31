"""A sentence-level segment: a run of words with an optional speaker label."""
from __future__ import annotations

from dataclasses import dataclass

from voxtral_client.constants import TOKEN_MS


@dataclass
class SegmentTimestamp:
    text: str
    start_token: int
    end_token: int
    speaker: str | None = None

    @property
    def start_s(self) -> float:
        return self.start_token * TOKEN_MS / 1000.0

    @property
    def end_s(self) -> float:
        return (self.end_token + 1) * TOKEN_MS / 1000.0
