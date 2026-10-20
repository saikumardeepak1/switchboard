"""Who is reading: a bearer token resolves to a principal scoped to one tenant.

`Authenticator` is pluggable so a deployment can verify JWTs or call its identity provider;
`StaticTokenAuthenticator` maps fixed tokens to principals, for tests and single-tenant setups.
Every activity and conversation read is scoped to the principal's tenant: another tenant's
conversations are invisible (empty lists, 404 for a direct read), not forbidden, so ids do not
leak by probing.
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Principal:
    tenant_id: str
    subject: str = ""


class Authenticator(Protocol):
    async def authenticate(self, token: str) -> Principal | None:
        """The principal a bearer token stands for, or None if the token is not valid."""
        ...


class StaticTokenAuthenticator:
    def __init__(self, tokens: Mapping[str, Principal]) -> None:
        self._tokens = dict(tokens)

    async def authenticate(self, token: str) -> Principal | None:
        found: Principal | None = None
        for known, principal in self._tokens.items():
            # Compare against every token so timing does not reveal which prefix matched.
            if hmac.compare_digest(known.encode(), token.encode()):
                found = principal
        return found
