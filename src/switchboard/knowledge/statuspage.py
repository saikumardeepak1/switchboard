"""Atlassian Statuspage v2 `incidents.json` ingestion (ADR-010).

Each incident becomes one fact per affected component (subject = component name, or "platform"
when none are listed), predicate "incident", valid from `created_at`. When a payload shows
`resolved_at`, the interval is closed there; history is kept by the store.

Transaction time defaults to the incident's `updated_at`, the moment the status page published
what we now know. That keeps `known_at` queries on the same epoch timeline as `as_of`, and makes
re-ingesting an unchanged payload a no-op. Pass `recorded_at` to stamp ingest time instead.

An incident that disappears from a later payload is left alone: `incidents.json` returns a
rolling window of recent incidents, so absence does not mean resolution.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from switchboard.knowledge.bitemporal import Fact, FactStore

PLATFORM_SUBJECT = "platform"
INCIDENT_PREDICATE = "incident"


def parse_timestamp(text: str) -> float:
    """ISO 8601 to epoch seconds. A timestamp without an offset is taken as UTC."""
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


class StatusPageIngestor:
    def __init__(self, store: FactStore, source: str) -> None:
        self._store = store
        self._source = source

    def ingest(self, payload: Mapping[str, Any], recorded_at: float | None = None) -> list[Fact]:
        """Apply a payload. Returns the fact versions that were written; [] when nothing changed.

        Raises ValueError on a payload that is not the incidents.json shape. Validation runs
        over every incident before any write, so a bad payload changes nothing.
        """
        incidents = payload.get("incidents")
        if not isinstance(incidents, list):
            raise ValueError("payload has no 'incidents' list")
        parsed = [self._parse(incident) for incident in incidents]

        written: list[Fact] = []
        for incident_id, subjects, value, valid_from, valid_to, updated_at in parsed:
            at = updated_at if recorded_at is None else recorded_at
            for subject in subjects:
                fact = self._apply(incident_id, subject, value, valid_from, valid_to, at)
                if fact is not None:
                    written.append(fact)
        return written

    def fact_id(self, incident_id: str, subject: str) -> str:
        return f"{self._source}:{incident_id}:{subject}"

    def _apply(
        self,
        incident_id: str,
        subject: str,
        value: dict[str, Any],
        valid_from: float,
        valid_to: float | None,
        at: float,
    ) -> Fact | None:
        fid = self.fact_id(incident_id, subject)
        existing = self._store.current(fid)
        if existing is not None:
            if existing.value == value and existing.valid_to == valid_to:
                return None
            # Never write behind the current version: a stale payload can arrive after a newer one.
            at = max(at, existing.recorded_at)
            if existing.value == value and existing.valid_to is None and valid_to is not None:
                return self._store.close(fid, valid_to, at)
        return self._store.assert_fact(
            subject,
            INCIDENT_PREDICATE,
            value,
            valid_from,
            valid_to,
            recorded_at=at,
            source=self._source,
            fact_id=fid,
        )

    @staticmethod
    def _parse(
        incident: Any,
    ) -> tuple[str, list[str], dict[str, Any], float, float | None, float]:
        if not isinstance(incident, Mapping):
            raise ValueError("incident is not an object")
        try:
            incident_id = str(incident["id"])
            name = str(incident["name"])
            created_at = parse_timestamp(incident["created_at"])
            updated_raw = incident.get("updated_at") or incident["created_at"]
            updated_at = parse_timestamp(updated_raw)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"incident is missing a required field: {exc}") from exc
        resolved_raw = incident.get("resolved_at")
        resolved_at = parse_timestamp(resolved_raw) if resolved_raw else None
        if resolved_at is not None and resolved_at <= created_at:
            # Statuspage allows backfilled incidents with zero duration; keep the interval valid.
            resolved_at = created_at + 1e-3

        components = incident.get("components") or []
        subjects = sorted(
            {str(c["name"]) for c in components if isinstance(c, Mapping) and c.get("name")}
        ) or [PLATFORM_SUBJECT]

        value = {
            "incident_id": incident_id,
            "name": name,
            "impact": str(incident.get("impact") or "none"),
            "status": str(incident.get("status") or ""),
            "update": _latest_update_body(incident.get("incident_updates") or []),
        }
        return incident_id, subjects, value, created_at, resolved_at, updated_at


def _latest_update_body(updates: list[Any]) -> str:
    """Body of the newest update. Statuspage lists newest first; sort anyway, stably, so an
    update with an unparseable timestamp falls back to list order."""
    best: tuple[float, int] | None = None
    body = ""
    for index, update in enumerate(updates):
        if not isinstance(update, Mapping):
            continue
        try:
            when = parse_timestamp(update["created_at"])
        except (KeyError, TypeError, ValueError):
            when = float("-inf")
        key = (when, -index)
        if best is None or key > best:
            best = key
            body = str(update.get("body") or "")
    return body
