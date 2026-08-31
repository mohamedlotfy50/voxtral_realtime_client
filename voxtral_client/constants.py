"""Voxtral DSM invariants (model-fixed, from the paper).

These describe the streaming tokenisation of the realtime model and are
not user-tunable; runtime/ hardware values live in :mod:`voxtral_client.config`.

  - frame_rate 12.5 Hz  ->  80ms per token
  - token_id 32  [STREAMING_PAD]   = silence / processing delay
  - token_id 33  [STREAMING_WORD]  = word boundary
  - token_id >= 1000               = word content
  - default delay 6 tokens  =  480ms
"""

TOKEN_MS: float = 80.0
STREAMING_PAD_ID: int = 32
STREAMING_WORD_ID: int = 33
SPECIAL_TOKEN_THRESHOLD: int = 1000
DEFAULT_DELAY_TOKENS: int = 6

PAUSE_BREAK_TOKENS: int = 12
