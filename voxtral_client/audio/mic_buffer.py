"""Thread-safe capped PCM buffer indexed by stream-start seconds.

Stores raw int16 bytes; :meth:`slice` returns float32 PCM for the requested
interval. The clock starts at 0 when :meth:`start` is called, which matches
the Voxtral token timeline (``token_position * 80ms`` from stream start).
"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from voxtral_client.config import Config


class MicRingBuffer:
    def __init__(self, config: "Config") -> None:
        self._config = config
        self.sample_rate = config.sample_rate
        self.max_bytes = config.ring_buffer_max_s * config.sample_rate * config.sample_width
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._t0: float | None = None
        self._total_bytes = 0

    def start(self) -> None:
        self._t0 = time.monotonic()

    def push_bytes(self, raw: bytes) -> None:
        with self._lock:
            self._buf.extend(raw)
            self._total_bytes += len(raw)
            if len(self._buf) > self.max_bytes:
                excess = len(self._buf) - self.max_bytes
                del self._buf[:excess]

    def slice(self, start_s: float, end_s: float) -> np.ndarray | None:
        """Return float32 PCM for ``[start_s, end_s]`` or ``None`` if unavailable."""
        if self._t0 is None:
            return None
        sw = self._config.sample_width
        start_sample = int(start_s * self.sample_rate)
        end_sample = int(end_s * self.sample_rate)
        if end_sample <= start_sample:
            return None
        with self._lock:
            global_start = start_sample * sw
            global_end = end_sample * sw
            kept_start = self._total_bytes - len(self._buf)
            if global_start < kept_start or global_end > self._total_bytes:
                return None
            lo = global_start - kept_start
            hi = global_end - kept_start
            chunk = bytes(self._buf[lo:hi])
        if not chunk:
            return None
        return np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
