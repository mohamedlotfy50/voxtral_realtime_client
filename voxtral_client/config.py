"""Runtime configuration loaded from ``config.yaml`` and injected into classes.

DSM model invariants (TOKEN_MS, token ids, delay) live in
:mod:`voxtral_client.constants` and are *not* part of ``Config`` — they are
fixed by the model, not by deployment. ``config.yaml`` may still list them
for documentation; this loader simply ignores keys it does not use.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class Config:
    served_model_name: str = "Voxtral-Mini-4B-Realtime-2602"
    api_key: str = "sk-local-voxtral-realtime"
    stt_host: str = "localhost"
    stt_port: int = 8000
    diarize_host: str = "localhost"
    diarize_port: int = 8001

    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2

    chunk_duration_s: float = 0.1
    arecord_block_samples: int = 256
    ring_buffer_max_s: int = 600

    # derived values (filled by __post_init__)
    chunk_samples: int = field(init=False, default=0)
    chunk_bytes: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.chunk_samples = int(self.sample_rate * self.chunk_duration_s)
        self.chunk_bytes = self.chunk_samples * self.sample_width

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data: dict = yaml.safe_load(f) or {}
        known = {f for f in cls.__dataclass_fields__ if cls.__dataclass_fields__[f].init}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)
