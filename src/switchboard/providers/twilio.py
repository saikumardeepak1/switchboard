"""Twilio Media Streams codec.

G.711 mu-law is implemented here in pure Python because `audioop` was removed in Python 3.13.
This is the common 16-bit variant (bias 0x84, clip 32635): a sample's biased magnitude picks a
segment (exponent 0..7) and a 4-bit mantissa, and decoding returns the midpoint of that
bucket, so the error is at most half a step, 2 ** (exponent + 2).

Twilio sends 8 kHz mono mu-law in 20 ms frames, 160 bytes each, base64 inside JSON messages.
"""

from __future__ import annotations

import base64
import binascii
import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

SAMPLE_RATE = 8000
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000
ULAW_SILENCE = 0xFF

_BIAS = 0x84
_CLIP = 32635


def linear_to_ulaw(sample: int) -> int:
    if not -32768 <= sample <= 32767:
        raise ValueError(f"sample out of 16-bit range: {sample}")
    sign = 0x80 if sample < 0 else 0
    magnitude = min(-sample if sample < 0 else sample, _CLIP) + _BIAS
    exponent = magnitude.bit_length() - 8
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def ulaw_to_linear(code: int) -> int:
    if not 0 <= code <= 0xFF:
        raise ValueError(f"mu-law code out of range: {code}")
    u = ~code & 0xFF
    magnitude = ((((u & 0x0F) << 3) + _BIAS) << ((u & 0x70) >> 4)) - _BIAS
    return -magnitude if u & 0x80 else magnitude


_DECODE = tuple(ulaw_to_linear(code) for code in range(256))
_ENCODE: bytes | None = None


def _encode_table() -> bytes:
    # 64 KiB table indexed by the unsigned 16-bit sample, built on first use.
    global _ENCODE
    if _ENCODE is None:
        _ENCODE = bytes(linear_to_ulaw(i - 65536 if i >= 32768 else i) for i in range(65536))
    return _ENCODE


def encode_ulaw(pcm: bytes) -> bytes:
    """16-bit signed little-endian PCM to mu-law bytes."""
    if len(pcm) % 2:
        raise ValueError("PCM16 data must have an even number of bytes")
    table = _encode_table()
    return bytes(table[s] for (s,) in struct.iter_unpack("<H", pcm))


def decode_ulaw(data: bytes) -> bytes:
    """Mu-law bytes to 16-bit signed little-endian PCM."""
    return struct.pack(f"<{len(data)}h", *(_DECODE[b] for b in data))


def split_frames(ulaw: bytes) -> list[bytes]:
    """Cut mu-law audio into 20 ms frames, padding the last one with silence."""
    frames = [ulaw[i : i + FRAME_BYTES] for i in range(0, len(ulaw), FRAME_BYTES)]
    if frames and len(frames[-1]) < FRAME_BYTES:
        frames[-1] = frames[-1] + bytes([ULAW_SILENCE]) * (FRAME_BYTES - len(frames[-1]))
    return frames


class TwilioProtocolError(ValueError):
    """An inbound message that is not valid Media Streams JSON."""


@dataclass(frozen=True, slots=True)
class TwilioConnected:
    protocol: str
    version: str


@dataclass(frozen=True, slots=True)
class TwilioStart:
    sequence_number: int
    stream_sid: str
    account_sid: str
    call_sid: str
    tracks: tuple[str, ...]
    custom_parameters: Mapping[str, str] = field(default_factory=dict)
    encoding: str = "audio/x-mulaw"
    sample_rate: int = SAMPLE_RATE
    channels: int = 1


@dataclass(frozen=True, slots=True)
class TwilioMedia:
    sequence_number: int
    stream_sid: str
    track: str
    chunk: int
    timestamp_ms: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class TwilioMark:
    sequence_number: int
    stream_sid: str
    name: str


@dataclass(frozen=True, slots=True)
class TwilioStop:
    sequence_number: int
    stream_sid: str
    account_sid: str
    call_sid: str


@dataclass(frozen=True, slots=True)
class TwilioDtmf:
    sequence_number: int
    stream_sid: str
    digit: str


TwilioMessage = TwilioConnected | TwilioStart | TwilioMedia | TwilioMark | TwilioStop | TwilioDtmf


def parse_message(raw: str | bytes) -> TwilioMessage:
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TwilioProtocolError(f"not JSON: {exc}") from exc
    if not isinstance(msg, dict):
        raise TwilioProtocolError("message is not a JSON object")
    event = msg.get("event")
    try:
        if event == "connected":
            return TwilioConnected(protocol=str(msg["protocol"]), version=str(msg["version"]))
        seq = int(msg["sequenceNumber"])
        sid = str(msg["streamSid"])
        if event == "start":
            start = _obj(msg, "start")
            fmt = _obj(start, "mediaFormat") if "mediaFormat" in start else {}
            return TwilioStart(
                sequence_number=seq,
                stream_sid=sid,
                account_sid=str(start["accountSid"]),
                call_sid=str(start["callSid"]),
                tracks=tuple(str(t) for t in start.get("tracks", ())),
                custom_parameters={
                    str(k): str(v) for k, v in dict(start.get("customParameters", {})).items()
                },
                encoding=str(fmt.get("encoding", "audio/x-mulaw")),
                sample_rate=int(fmt.get("sampleRate", SAMPLE_RATE)),
                channels=int(fmt.get("channels", 1)),
            )
        if event == "media":
            media = _obj(msg, "media")
            return TwilioMedia(
                sequence_number=seq,
                stream_sid=sid,
                track=str(media.get("track", "inbound")),
                chunk=int(media["chunk"]),
                timestamp_ms=int(media["timestamp"]),
                payload=base64.b64decode(media["payload"], validate=True),
            )
        if event == "mark":
            return TwilioMark(
                sequence_number=seq, stream_sid=sid, name=str(_obj(msg, "mark")["name"])
            )
        if event == "stop":
            stop = _obj(msg, "stop")
            return TwilioStop(
                sequence_number=seq,
                stream_sid=sid,
                account_sid=str(stop["accountSid"]),
                call_sid=str(stop["callSid"]),
            )
        if event == "dtmf":
            return TwilioDtmf(
                sequence_number=seq, stream_sid=sid, digit=str(_obj(msg, "dtmf")["digit"])
            )
    except (KeyError, TypeError, ValueError, binascii.Error) as exc:
        raise TwilioProtocolError(f"malformed {event!r} message: {exc!r}") from exc
    raise TwilioProtocolError(f"unknown event {event!r}")


def _obj(msg: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = msg[key]
    if not isinstance(value, dict):
        raise TypeError(f"{key!r} is not an object")
    return value


def build_media(stream_sid: str, ulaw: bytes) -> str:
    payload = base64.b64encode(ulaw).decode("ascii")
    return json.dumps({"event": "media", "streamSid": stream_sid, "media": {"payload": payload}})


def build_mark(stream_sid: str, name: str) -> str:
    """Twilio echoes a mark back once the audio queued before it has played."""
    return json.dumps({"event": "mark", "streamSid": stream_sid, "mark": {"name": name}})


def build_clear(stream_sid: str) -> str:
    """Drops all queued outbound audio, used on barge-in."""
    return json.dumps({"event": "clear", "streamSid": stream_sid})
