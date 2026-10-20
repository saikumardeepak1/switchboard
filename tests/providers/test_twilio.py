import base64
import json
import struct

import pytest
from hypothesis import given
from hypothesis import strategies as st

from switchboard.providers.twilio import (
    FRAME_BYTES,
    TwilioConnected,
    TwilioDtmf,
    TwilioMark,
    TwilioMedia,
    TwilioProtocolError,
    TwilioStart,
    TwilioStop,
    build_clear,
    build_mark,
    build_media,
    decode_ulaw,
    encode_ulaw,
    linear_to_ulaw,
    parse_message,
    split_frames,
    ulaw_to_linear,
)


def pcm(*samples: int) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def test_known_values() -> None:
    assert linear_to_ulaw(0) == 0xFF
    assert linear_to_ulaw(32767) == 0x80
    assert linear_to_ulaw(-32768) == 0x00
    assert linear_to_ulaw(-1) == 0x7F
    assert ulaw_to_linear(0xFF) == 0
    assert ulaw_to_linear(0x7F) == 0
    assert ulaw_to_linear(0x80) == 32124
    assert ulaw_to_linear(0x00) == -32124
    assert ulaw_to_linear(0xFE) == 8  # segment 0, mantissa 1: midpoint of [4, 12)
    assert ulaw_to_linear(0xEF) == 132  # segment 1, mantissa 0
    assert encode_ulaw(pcm(0, 32767, -32768)) == bytes([0xFF, 0x80, 0x00])
    assert decode_ulaw(bytes([0xFF, 0x80])) == pcm(0, 32124)


def test_every_code_round_trips_except_negative_zero() -> None:
    for code in range(256):
        expected = 0xFF if code == 0x7F else code
        assert linear_to_ulaw(ulaw_to_linear(code)) == expected


def quantization_bound(x: int) -> int:
    clipped = min(abs(x), 32635)
    exponent = max(0, (clipped + 0x84).bit_length() - 8)
    return (1 << (exponent + 2)) + max(0, abs(x) - 32635)


@given(st.lists(st.integers(min_value=-32768, max_value=32767), min_size=1, max_size=400))
def test_decode_encode_error_within_mu_law_quantization(samples: list[int]) -> None:
    decoded = struct.unpack(f"<{len(samples)}h", decode_ulaw(encode_ulaw(pcm(*samples))))
    for x, y in zip(samples, decoded, strict=True):
        assert abs(x - y) <= quantization_bound(x)


def test_encode_rejects_odd_length() -> None:
    with pytest.raises(ValueError):
        encode_ulaw(b"\x00")


def test_split_frames_pads_last_frame_with_silence() -> None:
    frames = split_frames(bytes(range(200)))
    assert [len(f) for f in frames] == [FRAME_BYTES, FRAME_BYTES]
    assert frames[1][40:] == b"\xff" * 120
    assert FRAME_BYTES == 160


def test_parse_connected_start_media_mark_stop_dtmf() -> None:
    assert parse_message(
        '{"event":"connected","protocol":"Call","version":"1.0.0"}'
    ) == TwilioConnected(protocol="Call", version="1.0.0")
    start = parse_message(
        json.dumps(
            {
                "event": "start",
                "sequenceNumber": "1",
                "streamSid": "MZ1",
                "start": {
                    "accountSid": "AC1",
                    "streamSid": "MZ1",
                    "callSid": "CA1",
                    "tracks": ["inbound"],
                    "customParameters": {"tenant": "acme"},
                    "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
                },
            }
        )
    )
    assert start == TwilioStart(
        sequence_number=1,
        stream_sid="MZ1",
        account_sid="AC1",
        call_sid="CA1",
        tracks=("inbound",),
        custom_parameters={"tenant": "acme"},
        encoding="audio/x-mulaw",
        sample_rate=8000,
        channels=1,
    )
    payload = bytes([0xFF] * 160)
    media = parse_message(
        json.dumps(
            {
                "event": "media",
                "sequenceNumber": "3",
                "streamSid": "MZ1",
                "media": {
                    "track": "inbound",
                    "chunk": "2",
                    "timestamp": "40",
                    "payload": base64.b64encode(payload).decode(),
                },
            }
        )
    )
    assert media == TwilioMedia(
        sequence_number=3,
        stream_sid="MZ1",
        track="inbound",
        chunk=2,
        timestamp_ms=40,
        payload=payload,
    )
    assert parse_message(
        b'{"event":"mark","sequenceNumber":"4","streamSid":"MZ1","mark":{"name":"utt_1"}}'
    ) == TwilioMark(sequence_number=4, stream_sid="MZ1", name="utt_1")
    assert parse_message(
        '{"event":"stop","sequenceNumber":"5","streamSid":"MZ1","stop":{"accountSid":"AC1","callSid":"CA1"}}'
    ) == TwilioStop(sequence_number=5, stream_sid="MZ1", account_sid="AC1", call_sid="CA1")
    assert parse_message(
        '{"event":"dtmf","sequenceNumber":"6","streamSid":"MZ1","dtmf":{"track":"inbound_track","digit":"7"}}'
    ) == TwilioDtmf(sequence_number=6, stream_sid="MZ1", digit="7")


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        '{"event":"teleport"}',
        '{"event":"media","sequenceNumber":"1","streamSid":"MZ1"}',
        '{"event":"media","sequenceNumber":"1","streamSid":"MZ1","media":{"track":"inbound",'
        '"chunk":"1","timestamp":"0","payload":"!!!"}}',
    ],
)
def test_parse_rejects_malformed(raw: str) -> None:
    with pytest.raises(TwilioProtocolError):
        parse_message(raw)


def test_build_outbound_messages() -> None:
    media = json.loads(build_media("MZ1", b"\x01\x02"))
    assert media == {"event": "media", "streamSid": "MZ1", "media": {"payload": "AQI="}}
    assert json.loads(build_mark("MZ1", "utt_1")) == {
        "event": "mark",
        "streamSid": "MZ1",
        "mark": {"name": "utt_1"},
    }
    assert json.loads(build_clear("MZ1")) == {"event": "clear", "streamSid": "MZ1"}
