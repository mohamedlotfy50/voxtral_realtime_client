# Voxtral Realtime Client

A fully object-oriented realtime speech-to-text transcription client for the
patched Voxtral vLLM server (see the companion `voxtral_realtime_server`
repository). It captures audio from a microphone or streams a WAV file to the
server's `/v1/realtime` WebSocket endpoint and computes **exact word-level
timestamps** using the DSM heartbeat mechanism described in the paper
(arXiv:2602.11298).

## How it works

Voxtral's streaming model emits a fixed-rate token stream (frame rate
12.5 Hz → 80 ms per token) with special boundary tokens:

| Token id | Meaning                       |
|----------|-------------------------------|
| 32       | `[STREAMING_PAD]` — silence / processing delay |
| 33       | `[STREAMING_WORD]` — word boundary |
| ≥ 1000   | word content                  |

The patched server forwards raw `token_ids` alongside each `transcription.delta`
event; the client uses token positions to derive precise start/end times for
each word and sentence. The model auto-detects the spoken language from audio
— no language hint is needed.

## Project structure

```
voxtral_realtime_client/
├── main.py                      # RealtimeClientApp entry point
├── pyproject.toml               # dependencies + package metadata
├── config.example.yaml          # committed config template (copy to config.yaml)
├── config.yaml                  # runtime configuration (gitignored)
├── data/                        # sample WAV files
└── voxtral_client/
    ├── config.py                # Config dataclass (loaded from config.yaml, injected)
    ├── constants.py             # DSM invariants from the paper (80ms/token, token IDs, delay)
    ├── audio/                   # MicCapture, MicRingBuffer, AudioFileReader, AudioStreamer
    ├── model/                   # WordTimestamp, SegmentTimestamp, SegmentCollector, TimestampTracker
    ├── nlp/                     # LanguageDetector, SentenceTerminators
    ├── net/                     # RealtimeClient (WebSocket), DiarizationClient
    └── presentation/            # TranscriptionDisplay, TranscriptionReceiver, TranscriptionSession
```

The codebase is fully OOP — no global functions. Each layer has a single
responsibility:

- **`config`** — runtime values (`Config` dataclass) loaded from
  `config.yaml` and passed (dependency injection) into the classes that need
  them. DSM model invariants (fixed by the model, not user-tunable) live in
  `constants.py`.
- **`audio`** — capture (mic via `sounddevice` with an `arecord` fallback),
  a thread-safe ring buffer for diarization slicing, WAV loading, and an
  `AudioStreamer` ABC with `MicAudioStreamer` / `FileAudioStreamer` subclasses.
- **`model`** — timestamp dataclasses and the `TimestampTracker` that maps
  token streams to timestamped words (with a local `tekken.json` decoding
  fallback when server delta text is unavailable).
- **`nlp`** — language detection (`lingua`) and per-language sentence
  terminators (Arabic `؟`, Hindi `।`/`॥`, etc.).
- **`net`** — thin WebSocket client (`RealtimeClient`) for the realtime
  protocol and an async REST client (`DiarizationClient`) for the speaker
  diarization server.
- **`presentation`** — display rendering (`TranscriptionDisplay`, including
  ANSI cursor redraw for live relabeling), the event-dispatch loop
  (`TranscriptionReceiver`), and the end-to-end orchestrator
  (`TranscriptionSession`).

## Requirements

- Python ≥ 3.10
- `websockets`, `numpy`, `pyyaml` (required)
- `lingua-language-detector` (required for `--detect-language`)
- `aiohttp` (required for `--diarize`)
- `sounddevice` (optional — falls back to `arecord` on Linux)
- `mistral-common` (optional — local token decoding fallback used only when
  the server sends no delta text; requires `--model-path` with `tekken.json`)

Install everything:

```bash
pip install -e .
# or the pinned versions (verified on Python 3.10):
pip install -r requirements.txt
```

## Configuration

Runtime defaults live in `config.yaml` (gitignored — it contains your real
server addresses). The committed template is [`config.example.yaml`](config.example.yaml);
copy it to get started:

```bash
cp config.example.yaml config.yaml
# then edit config.yaml to point stt_host / diarize_host at your servers
```

```yaml
served_model_name: "Voxtral-Mini-4B-Realtime-2602"
api_key: "sk-local-voxtral-realtime"
diarize_host: "localhost"
diarize_port: 8001
stt_host: "localhost"
stt_port: 8000
sample_rate: 16000
channels: 1
sample_width: 2
chunk_duration_s: 0.1
arecord_block_samples: 256
ring_buffer_max_s: 600
```

CLI flags override config values where applicable (`--host`, `--port`,
`--api-key`, `--model`, `--diarize-host`, `--diarize-port`).

> **Note on sample audio:** `data/` is gitignored to keep the repo small.
> Drop your own `.wav` files there before running file-mode examples.

## Usage

Start the patched server first (see `voxtral_realtime_server`), then:

```bash
# File mode — plain streaming text
python main.py --audio_path data/CSX_Wikipedia_Detector_demo.wav

# File mode — word-level timestamps
python main.py --audio_path data/CSX_Wikipedia_Detector_demo.wav --timestamps true --segment word

# File mode — sentence-level timestamps with language detection
python main.py --audio_path data/CSX_Wikipedia_Detector_demo.wav \
    --timestamps true --segment segment --detect-language

# Live microphone
python main.py --mic --timestamps true --segment word

# With speaker diarization (requires the diarization server running)
python main.py --audio_path data/diar_test_2spk_v2.wav --diarize \
    --diarize-host <your-diarization-server-ip> --diarize-port 8001

# Debug: print raw token IDs per delta
python main.py --mic --show-tokens
```

### Display modes

| `--timestamps` | `--segment` | Output |
|----------------|------------|--------|
| `false` (default) | — | inline streaming text |
| `true` | `word` | per-word timestamp table |
| `true` | `segment` | per-sentence timestamp table (with speaker labels if `--diarize`) |

## Architecture notes

- **Dependency injection**: `Config` is loaded once and passed into classes;
  DSM invariants (from the paper) live in `constants.py` since they are fixed
  by the model.
- **Optional dependencies** (`lingua`, `aiohttp`) are imported hard inside
  their feature modules, which are only imported lazily when the
  corresponding flag is enabled — the app runs without them, and features
  fail fast when enabled but the dep is missing.
- **`sounddevice` fallback**: on Linux the client automatically falls back to
  `arecord` if `sounddevice` is not installed or fails to open a device.

## Reference

- Paper: arXiv:2602.11298 (DSM heartbeat word-timestamp mechanism)
- Server: `voxtral_realtime_server` (patched vLLM with `token_ids` in delta events)
