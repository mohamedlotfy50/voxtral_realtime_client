"""Orchestrates a realtime transcription session end-to-end.

Composes the realtime WebSocket client, an audio streamer (mic or file), the
timestamp tracker / segment collector, the optional diarization client, and
the transcription receiver, then drives the send + receive + stop tasks.
Replaces the former ``run()`` free function.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from voxtral_client.audio.audio_file import AudioFileReader
from voxtral_client.audio.mic_buffer import MicRingBuffer
from voxtral_client.audio.mic_capture import MicCapture
from voxtral_client.audio.streamer import FileAudioStreamer, MicAudioStreamer
from voxtral_client.constants import PAUSE_BREAK_TOKENS
from voxtral_client.model.segment_collector import SegmentCollector
from voxtral_client.model.timestamp_tracker import TimestampTracker
from voxtral_client.net.realtime_client import RealtimeClient
from voxtral_client.presentation.display import TranscriptionDisplay
from voxtral_client.presentation.receiver import TranscriptionReceiver

if TYPE_CHECKING:
    from voxtral_client.config import Config
    from voxtral_client.net.diarization_client import DiarizationClient


class TranscriptionSession:
    def __init__(
        self,
        config: "Config",
        *,
        language: str | None,
        model_path: str | None,
        audio_path: str | None,
        mic: bool,
        device: str | None,
        timestamps: bool,
        segment_mode: str,
        detect_language: bool,
        show_tokens: bool,
        diarize: bool,
        stop_event: asyncio.Event,
    ) -> None:
        self._config = config
        self._language = language
        self._model_path = model_path
        self._audio_path = audio_path
        self._mic = mic
        self._device = device
        self._timestamps = timestamps
        self._segment_mode = segment_mode
        self._detect_language = detect_language
        self._show_tokens = show_tokens
        self._diarize = diarize
        self._stop_event = stop_event
        self._mic_capture: MicCapture | None = None

    async def run(self) -> None:
        self._print_preamble()

        lang_det = self._maybe_enable_language_detection()
        self._maybe_force_diarization_mode()

        if self._timestamps:
            print(f"Mode: timestamps=true, segment={self._segment_mode}")
        else:
            print("Mode: plain streaming (no timestamps)")

        tracker = TimestampTracker(model_path=self._model_path)
        segment_collector: SegmentCollector | None = None
        if self._timestamps and self._segment_mode == "segment":
            segment_collector = SegmentCollector(
                language=self._language,
                pause_threshold_tokens=PAUSE_BREAK_TOKENS,
            )

        ring = self._maybe_build_ring()
        diar = await self._maybe_start_diarization()
        mic_capture = self._maybe_start_mic()

        try:
            await self._run_connection(tracker, segment_collector, ring, diar, lang_det)
        finally:
            if diar is not None:
                await diar.close()
            if mic_capture is not None:
                mic_capture.stop()

    async def _run_connection(self, tracker, segment_collector, ring, diar, lang_det):
        client = RealtimeClient(
            self._config.stt_host,
            self._config.stt_port,
            self._config.served_model_name,
            self._config.api_key,
        )
        async with client:
            session = await client.wait_session_created()
            if session is None:
                return
            print(f"Session: {session.get('id', '?')}")
            await client.send_session_update(self._config.served_model_name, self._language)
            await client.send_commit()

            streamer = self._build_streamer()
            if self._mic:
                print("Speak into the microphone (Ctrl+C to stop):\n")

            stream_task = asyncio.create_task(streamer.stream(client, ring))
            display = TranscriptionDisplay(self._detect_language, lang_det)
            receiver = TranscriptionReceiver(
                client=client,
                tracker=tracker,
                segment_collector=segment_collector,
                timestamps=self._timestamps,
                segment_mode=self._segment_mode,
                show_tokens=self._show_tokens,
                display=display,
                ring=ring,
                diar=diar,
            )
            recv_task = asyncio.create_task(receiver.run())
            stop_done = asyncio.create_task(self._stop_event.wait())

            await asyncio.wait(
                {stream_task, recv_task, stop_done},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not recv_task.done():
                try:
                    await client.send_commit(final=True)
                    await asyncio.wait_for(recv_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

            for t in (stream_task, recv_task):
                if not t.done():
                    t.cancel()

    def _build_streamer(self):
        if self._mic:
            return MicAudioStreamer(self._mic_capture)
        return FileAudioStreamer(self._audio_path, AudioFileReader(self._config))

    def _print_preamble(self) -> None:
        if self._language:
            print(f"Language hint: {self._language}")
        else:
            print("Language: auto-detect (model handles this internally)")

    def _maybe_enable_language_detection(self):
        if not self._detect_language:
            return None
        from voxtral_client.nlp.language_detection import LanguageDetector

        lang_det = LanguageDetector()
        print("Language detection: enabled")
        return lang_det

    def _maybe_force_diarization_mode(self) -> None:
        if not self._diarize:
            return
        self._timestamps = True
        self._segment_mode = "segment"

    def _maybe_build_ring(self) -> MicRingBuffer | None:
        if (
            self._timestamps
            and self._segment_mode == "segment"
            and (self._mic or self._audio_path)
        ):
            return MicRingBuffer(self._config)
        return None

    async def _maybe_start_diarization(self):
        if not self._diarize:
            return None
        from voxtral_client.net.diarization_client import DiarizationClient

        diar = DiarizationClient(self._config.diarize_host, self._config.diarize_port)
        print(f"Diarization: {self._config.diarize_host}:{self._config.diarize_port}")
        try:
            await diar.start()
            print(f"Diarization session: {diar.session_id}")
            return diar
        except Exception as e:
            print(f"[diarize] failed to start session: {e} -> disabling")
            await diar.close()
            return None

    def _maybe_start_mic(self) -> MicCapture | None:
        if not self._mic:
            return None
        self._mic_capture = MicCapture(self._device, self._config)
        self._mic_capture.start()
        return self._mic_capture
