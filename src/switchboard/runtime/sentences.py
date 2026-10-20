"""Split streamed text into sentences as it arrives, so each can be gated before TTS."""

from __future__ import annotations

import re

# Terminal punctuation, optional closing quotes or brackets, then whitespace. Requiring the
# whitespace keeps "$12.50" and "v2.1" whole.
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*\s+")


class SentenceSplitter:
    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        out: list[str] = []
        start = 0
        for match in _BOUNDARY.finditer(self._buffer):
            sentence = self._buffer[start : match.end()].strip()
            if sentence:
                out.append(sentence)
            start = match.end()
        self._buffer = self._buffer[start:]
        return out

    def flush(self) -> str | None:
        rest, self._buffer = self._buffer.strip(), ""
        return rest or None
