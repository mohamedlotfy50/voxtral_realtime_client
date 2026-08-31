"""Thin WebSocket client for the patched vLLM ``/v1/realtime`` endpoint.

Wraps connection lifecycle, the ``session.created`` handshake, and the
``input_audio_buffer.append`` / ``commit`` messages, exposing a small typed
surface so the rest of the app never touches the raw socket.
"""
from __future__ import annotations

import base64
import json
from typing import Any

import websockets


class RealtimeClient:
    def __init__(self, host: str, port: int, model: str, api_key: str | None) -> None:
        self._uri = f"ws://{host}:{port}/v1/realtime?model={model}"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._ws: websockets.WebSocketClientProtocol | None = None

    async def __aenter__(self) -> "RealtimeClient":
        print(f"Connecting to {self._uri} ...")
        self._ws = await websockets.connect(self._uri, additional_headers=self._headers)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def wait_session_created(self) -> dict | None:
        """Receive and validate the initial ``session.created`` event."""
        resp = await self.recv_json()
        if resp is None or resp.get("type") != "session.created":
            print(f"Unexpected: {resp}")
            return None
        return resp

    async def send_session_update(self, model: str, language: str | None) -> None:
        update: dict[str, Any] = {"type": "session.update", "model": model}
        if language:
            update["language"] = language
        await self._send_json(update)

    async def send_audio_chunk(self, chunk: bytes) -> None:
        await self._send_json({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("utf-8"),
        })

    async def send_commit(self, final: bool = False) -> None:
        msg: dict[str, Any] = {"type": "input_audio_buffer.commit"}
        if final:
            msg["final"] = True
        await self._send_json(msg)

    async def recv_json(self, timeout: float = 0.5) -> dict | None:
        """Receive one JSON message; ``None`` on timeout."""
        import asyncio

        assert self._ws is not None
        try:
            msg = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return json.loads(msg)

    async def send_raw(self, message: str) -> None:
        assert self._ws is not None
        await self._ws.send(message)

    async def _send_json(self, obj: dict) -> None:
        assert self._ws is not None
        await self._ws.send(json.dumps(obj))
