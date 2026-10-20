"""Who a Twilio media stream belongs to, decided by the server and never by the caller.

The WebSocket upgrade carries nothing the server issued, and `customParameters` on the `start`
message come from whatever opened the socket. So tenant, customer and party are read only from a
stream token the server signed when it wrote the TwiML for the call: `<Parameter name="token">`.

Token format: `base64url(json claims) "." base64url(HMAC-SHA256(secret, first part))`. Claims are
`tenant`, `customer`, `party`, and optionally `exp` (clock seconds) and `call` (the call SID the
token was issued for, checked against the stream's `callSid`).

Optionally the upgrade request is also checked against Twilio's `X-Twilio-Signature`: base64 of
HMAC-SHA1 under the account auth token over the full URL, followed by each POST parameter name
and value sorted by name (a WebSocket upgrade has none). A stream that fails either check never
reaches the session factory.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass

from switchboard.core import Clock

# WebSocket close code for a policy violation (RFC 6455).
POLICY_VIOLATION = 1008


class InvalidStreamToken(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StreamClaims:
    tenant: str
    customer: str
    party: str = "caller"

    def parameters(self) -> dict[str, str]:
        return {"tenant": self.tenant, "customer": self.customer, "party": self.party}


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError) as exc:
        raise InvalidStreamToken("token is not base64url") from exc


class StreamTokenSigner:
    def __init__(self, secret: bytes, *, clock: Clock | None = None) -> None:
        if len(secret) < 16:
            raise ValueError("stream token secret must be at least 16 bytes")
        self._secret = secret
        self._clock = clock

    def _sign(self, body: str) -> str:
        return _b64(hmac.new(self._secret, body.encode(), hashlib.sha256).digest())

    def issue(
        self,
        *,
        tenant: str,
        customer: str,
        party: str = "caller",
        call_sid: str | None = None,
        ttl_s: float | None = None,
    ) -> str:
        claims: dict[str, object] = {"tenant": tenant, "customer": customer, "party": party}
        if call_sid is not None:
            claims["call"] = call_sid
        if ttl_s is not None:
            if self._clock is None:
                raise ValueError("an expiring token needs a clock")
            claims["exp"] = self._clock.now() + ttl_s
        body = _b64(json.dumps(claims, sort_keys=True, separators=(",", ":")).encode())
        return f"{body}.{self._sign(body)}"

    def verify(self, token: str, *, call_sid: str | None = None) -> StreamClaims:
        parts = token.split(".")
        if len(parts) != 2 or not all(parts):
            raise InvalidStreamToken("malformed token")
        body, signature = parts
        if not hmac.compare_digest(self._sign(body), signature):
            raise InvalidStreamToken("bad signature")
        try:
            claims = json.loads(_unb64(body))
        except ValueError as exc:
            raise InvalidStreamToken("token claims are not JSON") from exc
        if not isinstance(claims, dict):
            raise InvalidStreamToken("token claims are not an object")
        tenant, customer = claims.get("tenant"), claims.get("customer")
        party = claims.get("party", "caller")
        if not all(isinstance(v, str) and v for v in (tenant, customer, party)):
            raise InvalidStreamToken("token needs tenant, customer and party")
        exp = claims.get("exp")
        if exp is not None:
            if self._clock is None or not isinstance(exp, int | float):
                raise InvalidStreamToken("token expiry cannot be checked")
            if self._clock.now() >= exp:
                raise InvalidStreamToken("token expired")
        bound = claims.get("call")
        if bound is not None and bound != call_sid:
            raise InvalidStreamToken("token was issued for another call")
        assert isinstance(tenant, str) and isinstance(customer, str) and isinstance(party, str)
        return StreamClaims(tenant, customer, party)


def twilio_signature(auth_token: str, url: str, params: Mapping[str, str] | None = None) -> str:
    """Twilio's request signature: base64 HMAC-SHA1 over the URL plus sorted name+value pairs."""
    data = url + "".join(f"{k}{v}" for k, v in sorted((params or {}).items()))
    digest = hmac.new(auth_token.encode(), data.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


@dataclass(frozen=True)
class StreamAuth:
    """How the media endpoint authenticates a stream. The signed token is always required;
    `twilio_auth_token` adds the `X-Twilio-Signature` check on the upgrade request."""

    signer: StreamTokenSigner
    twilio_auth_token: str | None = None
    token_parameter: str = "token"

    def check_upgrade(self, url: str, signature: str | None) -> bool:
        if self.twilio_auth_token is None:
            return True
        if not signature:
            return False
        expected = twilio_signature(self.twilio_auth_token, url)
        return hmac.compare_digest(expected, signature)

    def claims(self, call_sid: str, custom_parameters: Mapping[str, str]) -> StreamClaims:
        token = custom_parameters.get(self.token_parameter)
        if not token:
            raise InvalidStreamToken("no stream token")
        return self.signer.verify(token, call_sid=call_sid)
