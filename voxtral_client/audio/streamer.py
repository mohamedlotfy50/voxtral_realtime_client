"""Audio sources that stream PCM chunks to the realtime server.

``AudioStreamer`` is the abstract base; the two concrete implementations are
microphone capture (live, open-ended) and file streaming (finite, sends a
final commit when the file is exhausted).
"""
from __future__ import annotations

import asyncio
import base64
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from voxtral_client.audio.audio_file import AudioFileReader
from voxtral_client.audio.mic_capture import MicCapture

if TYPE_CHECKING:
    from voxtral_client.audio.mic_buffer import MicRingBuffer
    from voxtral_client.net.realtime_client import RealtimeClient


class AudioStreamer(ABC):
    @abstractmethod
    async def stream(
        self, client: "RealtimeClient", ring: "MicRingBuffer | None"
    ) -> None:
        """Push audio chunks to the server until exhausted or cancelled."""


class MicAudioStreamer(AudioStreamer):
    def __init__(self, mic_capture: MicCapture) -> None:
        self._mic = mic_capture

    async def stream(
        self, client: "RealtimeClient", ring: "MicRingBuffer | None"
    ) -> None:
        loop = asyncio.get_event_loop()
        if ring is not None:
            ring.start()
        while True:
            chunk = await loop.run_in_executor(None, self._mic.read_chunk)
            if not chunk:
                raise RuntimeError("mic stream closed")
            if ring is not None:
                ring.push_bytes(chunk)
            await client.send_audio_chunk(chunk)


class FileAudioStreamer(AudioStreamer):
    def __init__(self, audio_path: str, audio_reader: AudioFileReader) -> None:
        self._audio_path = audio_path
        self._reader = audio_reader

    async def stream(
        self, client: "RealtimeClient", ring: "MicRingBuffer | None"
    ) -> None:
        audio_bytes = self._reader.load_pcm16(self._audio_path)
        total = self._reader.total_chunks(audio_bytes)
        print(f"Sending {total} audio chunks ({len(audio_bytes)} bytes)...")
        if ring is not None:
            ring.start()
        for chunk in self._reader.iter_chunks(audio_bytes):
            if ring is not None:
                ring.push_bytes(chunk)
            await client.send_audio_chunk(chunk)
            await asyncio.sleep(0.01)
        await client.send_commit(final=True)
        print("Audio sent. Waiting for transcription...\n")
