"""Loads a WAV file and exposes it as 16kHz mono PCM16 bytes/chunks."""
from __future__ import annotations

import wave
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from voxtral_client.config import Config


class AudioFileReader:
    def __init__(self, config: "Config") -> None:
        self._config = config

    def load_pcm16(self, audio_path: str) -> bytes:
        """Load a WAV file and convert to PCM16 @ 16kHz mono bytes."""
        with wave.open(audio_path, "rb") as w:
            n_channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            frames = w.readframes(w.getnframes())

        if sampwidth != 2:
            raise ValueError(f"Expected 16-bit WAV, got sample width {sampwidth}")

        audio = np.frombuffer(frames, dtype=np.int16)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

        if framerate != self._config.sample_rate:
            ratio = framerate // self._config.sample_rate
            if framerate != self._config.sample_rate * ratio:
                raise ValueError(
                    f"Cannot downsample {framerate}Hz to "
                    f"{self._config.sample_rate}Hz"
                )
            audio = audio[::ratio]

        return audio.tobytes()

    def iter_chunks(self, audio_bytes: bytes) -> Iterator[bytes]:
        step = self._config.chunk_bytes
        for i in range(0, len(audio_bytes), step):
            yield audio_bytes[i : i + step]

    def total_chunks(self, audio_bytes: bytes) -> int:
        return (len(audio_bytes) + self._config.chunk_bytes - 1) // self._config.chunk_bytes
