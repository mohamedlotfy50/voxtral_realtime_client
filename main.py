#!/usr/bin/env python3
"""Entry point for the Voxtral realtime transcription client.

Parses CLI options, loads ``config.yaml``, wires signal handling, and runs a
:class:`TranscriptionSession`. Configuration defaults come from
``config.yaml``; CLI flags override them where applicable.
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from dataclasses import replace

from voxtral_client.config import Config
from voxtral_client.presentation.session import TranscriptionSession


class RealtimeClientApp:
    def __init__(self, argv: list[str] | None = None) -> None:
        self._argv = argv

    def run(self) -> None:
        args = self._parse_args()
        config = self._build_config(args)

        stop_event = asyncio.Event()

        def _stop(*_):
            stop_event.set()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _stop)
            except NotImplementedError:
                signal.signal(sig, _stop)

        try:
            loop.run_until_complete(
                TranscriptionSession(
                    config,
                    language=args.language,
                    model_path=args.model_path,
                    audio_path=args.audio_path,
                    mic=args.mic,
                    device=args.device,
                    timestamps=args.timestamps == "true",
                    segment_mode=args.segment,
                    detect_language=args.detect_language,
                    show_tokens=args.show_tokens,
                    diarize=args.diarize,
                    stop_event=stop_event,
                ).run()
            )
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()

    def _parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Realtime transcription with word-level timestamps"
        )
        parser.add_argument(
            "--config", default="config.yaml",
            help="Path to the YAML config file (default: %(default)s).",
        )
        parser.add_argument(
            "--model", default=None,
            help="Override served model name (default: from config).",
        )
        parser.add_argument(
            "--model-path", default=None,
            help="Optional: path to Voxtral model dir (for local token decoding "
                 "fallback). Not required — the server provides decoded text.",
        )
        parser.add_argument(
            "--host", default=None,
            help="Override STT server host (default: from config).",
        )
        parser.add_argument(
            "--port", type=int, default=None,
            help="Override STT server port (default: from config).",
        )
        parser.add_argument(
            "--api-key", default=None,
            help="Override API key (default: from config).",
        )
        parser.add_argument(
            "--language", type=str, default=None,
            help="ISO-639-1 language hint (default: None = auto-detect). "
                 "Supported: en, fr, es, de, it, pt, nl, ar, hi",
        )
        parser.add_argument(
            "--detect-language", action="store_true",
            help="Display detected language per word/segment (e.g. [EN], [FR]). "
                 "Requires: pip install lingua-language-detector",
        )
        parser.add_argument(
            "--audio_path", type=str, default=None,
            help="Path to WAV file to transcribe (file mode).",
        )
        parser.add_argument(
            "--mic", action="store_true",
            help="Use microphone input (live mode).",
        )
        parser.add_argument(
            "--device", type=str, default=None,
            help="Audio device. Linux: ALSA device (e.g. plughw:3,0) or index. "
                 "Defaults to system default.",
        )
        parser.add_argument(
            "--show-tokens", action="store_true",
            help="Print raw token IDs for each delta (debug).",
        )
        parser.add_argument(
            "--timestamps", type=str, default="false",
            choices=["true", "false"],
            help="Enable timestamp display (default: false = plain streaming).",
        )
        parser.add_argument(
            "--segment", type=str, default="word",
            choices=["word", "segment"],
            help="Timestamp granularity when --timestamps=true (default: word).",
        )
        parser.add_argument(
            "--diarize", action="store_true",
            help="Enable speaker diarization (requires the diarization server). "
                 "Forces timestamps=true, segment=segment.",
        )
        parser.add_argument(
            "--diarize-host", default=None,
            help="Override diarization server host (default: from config).",
        )
        parser.add_argument(
            "--diarize-port", type=int, default=None,
            help="Override diarization server port (default: from config).",
        )
        args = parser.parse_args(self._argv)

        if not args.mic and not args.audio_path:
            parser.print_help()
            print("\nERROR: Provide --mic or --audio_path")
            sys.exit(1)
        return args

    def _build_config(self, args: argparse.Namespace) -> Config:
        config = Config.from_yaml(args.config)
        overrides: dict = {}
        if args.model is not None:
            overrides["served_model_name"] = args.model
        if args.host is not None:
            overrides["stt_host"] = args.host
        if args.port is not None:
            overrides["stt_port"] = args.port
        if args.api_key is not None:
            overrides["api_key"] = args.api_key
        if args.diarize_host is not None:
            overrides["diarize_host"] = args.diarize_host
        if args.diarize_port is not None:
            overrides["diarize_port"] = args.diarize_port
        return replace(config, **overrides) if overrides else config


if __name__ == "__main__":
    RealtimeClientApp().run()
