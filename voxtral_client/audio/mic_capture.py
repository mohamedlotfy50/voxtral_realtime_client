"""Cross-platform microphone capture using ``sounddevice``.

Falls back to ``arecord`` on Linux when ``sounddevice`` is not available or
fails to open a device. Chunks are ``config.chunk_bytes`` long.
"""
from __future__ import annotations

import platform
import queue
import subprocess
import sys
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voxtral_client.config import Config


class MicCapture:
    def __init__(self, device: str | None, config: "Config") -> None:
        self._config = config
        self.device = device
        self._sd = None
        self._stream = None
        self._arecord_proc: subprocess.Popen | None = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._use_arecord = False

    def start(self) -> None:
        try:
            import sounddevice as sd

            self._sd = sd
            sd.default.samplerate = self._config.sample_rate
            sd.default.channels = self._config.channels
            sd.default.dtype = "int16"

            device = self._resolve_device()
            self._stream = sd.RawInputStream(
                samplerate=self._config.sample_rate,
                blocksize=self._config.chunk_samples,
                dtype="int16",
                channels=self._config.channels,
                device=device,
                callback=self._sd_callback,
            )
            self._stream.start()
            print(f"[sounddevice] Mic started (device={device or 'default'})")
        except ImportError:
            if platform.system() == "Linux":
                print("[sounddevice not installed, falling back to arecord]")
                self._use_arecord = True
                self._start_arecord()
            else:
                raise RuntimeError(
                    "sounddevice is not installed. Install it with: "
                    "pip install sounddevice"
                )
        except Exception as e:
            if platform.system() == "Linux":
                print(f"[sounddevice failed: {e}, falling back to arecord]")
                self._use_arecord = True
                self._start_arecord()
            else:
                raise

    def read_chunk(self) -> bytes | None:
        """Read one audio chunk; ``None`` if the stream ended."""
        chunk_bytes = self._config.chunk_bytes
        if self._use_arecord:
            assert self._arecord_proc is not None
            raw = self._arecord_proc.stdout.read(chunk_bytes)
            return raw if raw else None

        buf = bytearray()
        while len(buf) < chunk_bytes:
            try:
                piece = self._audio_queue.get(timeout=5.0)
                buf.extend(piece)
            except queue.Empty:
                if buf:
                    break
                return None
        return bytes(buf[:chunk_bytes])

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._arecord_proc:
            self._arecord_proc.terminate()
            try:
                self._arecord_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._arecord_proc.kill()
            self._arecord_proc = None

    def _resolve_device(self):
        if self.device and self.device.isdigit():
            return int(self.device)
        return None

    def _sd_callback(self, indata, frames, time_info, status):
        if status:
            sys.stderr.write(f"[sounddevice] {status}\n")
        self._audio_queue.put(bytes(indata))

    def _start_arecord(self) -> None:
        cmd = [
            "arecord", "-q", "-t", "raw", "-f", "S16_LE",
            "-c", str(self._config.channels),
            "-r", str(self._config.sample_rate),
        ]
        if self.device and not self.device.isdigit():
            cmd.extend(["-D", self.device])
        self._arecord_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self._config.chunk_bytes * 4,
        )
