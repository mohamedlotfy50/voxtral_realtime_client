"""Async REST client for the speaker-diarization server.

``aiohttp`` is imported hard per project policy: enabling ``--diarize`` fails
fast if the dependency is not installed.
"""
from __future__ import annotations

import base64

import aiohttp
import numpy as np


class DiarizationClient:
    def __init__(self, host: str, port: int) -> None:
        self.base = f"http://{host}:{port}/v1/diarization"
        self.session_id: str | None = None
        self._http: aiohttp.ClientSession | None = None
        self.prev_speaker: int | None = None

    async def start(self) -> None:
        self._http = aiohttp.ClientSession()
        async with self._http.post(f"{self.base}/sessions") as r:
            r.raise_for_status()
            self.session_id = (await r.json())["session_id"]

    async def label(self, pcm: np.ndarray, start_s: float, end_s: float) -> dict:
        assert self._http is not None and self.session_id is not None
        audio_b64 = base64.b64encode(
            (pcm * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()
        ).decode("utf-8")
        payload = {
            "audio": audio_b64,
            "start_s": start_s,
            "end_s": end_s,
            "prev_speaker": self.prev_speaker,
        }
        async with self._http.post(
            f"{self.base}/sessions/{self.session_id}/label", json=payload
        ) as r:
            r.raise_for_status()
            res = await r.json()
        self.prev_speaker = res["speaker_idx"]
        return res

    async def consolidate(self) -> dict:
        """Force a final consolidation; returns ``{relabel, num_speakers, speakers}``."""
        assert self._http is not None and self.session_id is not None
        async with self._http.post(
            f"{self.base}/sessions/{self.session_id}/consolidate"
        ) as r:
            r.raise_for_status()
            return await r.json()

    async def summary(self) -> dict:
        assert self._http is not None and self.session_id is not None
        async with self._http.get(
            f"{self.base}/sessions/{self.session_id}"
        ) as r:
            r.raise_for_status()
            return await r.json()

    async def close(self) -> None:
        if self._http is not None and self.session_id is not None:
            try:
                await self._http.delete(
                    f"{self.base}/sessions/{self.session_id}"
                )
            except Exception:
                pass
        if self._http is not None:
            await self._http.close()
            self._http = None
