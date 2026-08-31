"""A single timestamped word produced by the DSM heartbeat mechanism."""
from __future__ import annotations

from dataclasses import dataclass

from voxtral_client.constants import TOKEN_MS


@dataclass
class WordTimestamp:
    text: str
    start_token: int
    end_token: int
    preceding_silence_tokens: int = 0

    @property
    def start_s(self) -> float:
        return self.start_token * TOKEN_MS / 1000.0

    @property
    def end_s(self) -> float:
        return (self.end_token + 1) * TOKEN_MS / 1000.0
