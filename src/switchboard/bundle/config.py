"""`agent.yaml`: the agent's identity, prompt, voice, routing, limits and parties.

The bundle sits below `brain` and `knowledge` in the import order, so router overrides and
knowledge documents are plain data here; the runtime turns them into brain and knowledge types.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from switchboard.bundle.runtime_tools import RUNTIME_TOOLS

TierName = Literal["talker", "thinker"]
_TIERS = ("talker", "thinker")

_TOP_KEYS = frozenset(
    {
        "name",
        "description",
        "system_prompt",
        "voice",
        "router",
        "eager_threshold",
        "tenant_limits",
        "max_policy_retries",
        "handoff_message",
        "runtime_tools",
        "parties",
        "rollback_on",
        "bridge_phrases",
        "bridge_after_ms",
    }
)
_VOICE_KEYS = frozenset({"voice_id", "ttfb_s", "chars_per_s"})
_ROUTER_KEYS = frozenset({"default_tier", "rules"})
_RULE_KEYS = frozenset(
    {"name", "tier", "patterns", "state_flag", "min_words", "max_words", "min_questions"}
)
_LIMIT_KEYS = frozenset({"max_concurrent_conversations"})
_PARTY_KEYS = frozenset({"description", "confirms"})

DEFAULT_HANDOFF_MESSAGE = "I can't finish that myself, so I'm bringing in a teammate."


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class VoiceSettings:
    voice_id: str = "default"
    ttfb_s: float = 0.2
    chars_per_s: float = 15.0


@dataclass(frozen=True)
class RouterRuleSpec:
    """Mirrors `brain.RouteRule`. Bundle rules run before the runtime's default rules."""

    name: str
    tier: TierName
    patterns: tuple[str, ...] = ()
    state_flag: str | None = None
    min_words: int | None = None
    max_words: int | None = None
    min_questions: int | None = None


@dataclass(frozen=True)
class TenantLimits:
    max_concurrent_conversations: int = 10


@dataclass(frozen=True)
class PartyConfig:
    """A party the agent may call with `call_party`.

    `confirms` names the evidence flag a child call's `confirmed` answer sets. It is fixed in the
    bundle so the model can never choose which flag a call result writes.
    """

    name: str
    description: str = ""
    confirms: str | None = None


@dataclass(frozen=True)
class AgentConfig:
    name: str
    description: str
    system_prompt: str
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    router_default_tier: TierName = "thinker"
    router_rules: tuple[RouterRuleSpec, ...] = ()
    eager_threshold: float = 0.5
    tenant_limits: TenantLimits = field(default_factory=TenantLimits)
    max_policy_retries: int = 2
    handoff_message: str = DEFAULT_HANDOFF_MESSAGE
    runtime_tools: tuple[str, ...] = ()
    parties: Mapping[str, PartyConfig] = field(default_factory=dict)
    # Conversation outcomes (and "turn_aborted") after which the runtime compensates writes.
    rollback_on: tuple[str, ...] = ()
    # Fixed phrases spoken while a tool runs longer than `bridge_after_ms` (ADR-008).
    bridge_phrases: tuple[str, ...] = ()
    bridge_after_ms: int = 700


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping")
    return value


def _only(data: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    for key in data:
        if key not in allowed:
            raise ConfigError(f"{where}: unknown key {key!r}")


def _str(data: Mapping[str, Any], key: str, where: str, default: str | None = None) -> str:
    value = data.get(key, default)
    if value is None:
        raise ConfigError(f"{where}: missing {key!r}")
    if not isinstance(value, str):
        raise ConfigError(f"{where}: {key!r} must be a string")
    return value.strip() if key == "system_prompt" else value


def _number(data: Mapping[str, Any], key: str, where: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{where}: {key!r} must be a number")
    return float(value)


def _int(data: Mapping[str, Any], key: str, where: str, default: int | None) -> int | None:
    value = data.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}: {key!r} must be an integer")
    return value


def _tier(value: object, where: str) -> TierName:
    if value == "talker":
        return "talker"
    if value == "thinker":
        return "thinker"
    raise ConfigError(f"{where}: tier must be one of {_TIERS}, got {value!r}")


def _voice(raw: object) -> VoiceSettings:
    data = _mapping(raw, "agent.yaml: 'voice'")
    _only(data, _VOICE_KEYS, "agent.yaml voice")
    voice = VoiceSettings(
        voice_id=_str(data, "voice_id", "agent.yaml voice", VoiceSettings.voice_id),
        ttfb_s=_number(data, "ttfb_s", "agent.yaml voice", VoiceSettings.ttfb_s),
        chars_per_s=_number(data, "chars_per_s", "agent.yaml voice", VoiceSettings.chars_per_s),
    )
    if voice.ttfb_s < 0 or voice.chars_per_s <= 0:
        raise ConfigError("agent.yaml voice: need ttfb_s >= 0 and chars_per_s > 0")
    return voice


def _router(raw: object) -> tuple[TierName, tuple[RouterRuleSpec, ...]]:
    data = _mapping(raw, "agent.yaml: 'router'")
    _only(data, _ROUTER_KEYS, "agent.yaml router")
    default = _tier(data.get("default_tier", "thinker"), "agent.yaml router.default_tier")
    raw_rules = data.get("rules") or []
    if not isinstance(raw_rules, list):
        raise ConfigError("agent.yaml router.rules must be a list")
    rules: list[RouterRuleSpec] = []
    for i, item in enumerate(raw_rules):
        where = f"agent.yaml router.rules[{i}]"
        rule = _mapping(item, where)
        _only(rule, _RULE_KEYS, where)
        patterns = rule.get("patterns") or []
        if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
            raise ConfigError(f"{where}: patterns must be a list of strings")
        flag = rule.get("state_flag")
        if flag is not None and not isinstance(flag, str):
            raise ConfigError(f"{where}: state_flag must be a string")
        rules.append(
            RouterRuleSpec(
                name=_str(rule, "name", where),
                tier=_tier(rule.get("tier"), where),
                patterns=tuple(patterns),
                state_flag=flag,
                min_words=_int(rule, "min_words", where, None),
                max_words=_int(rule, "max_words", where, None),
                min_questions=_int(rule, "min_questions", where, None),
            )
        )
    return default, tuple(rules)


def _limits(raw: object) -> TenantLimits:
    data = _mapping(raw, "agent.yaml: 'tenant_limits'")
    _only(data, _LIMIT_KEYS, "agent.yaml tenant_limits")
    cap = _int(
        data,
        "max_concurrent_conversations",
        "agent.yaml tenant_limits",
        TenantLimits.max_concurrent_conversations,
    )
    if cap is None or cap < 1:
        raise ConfigError("agent.yaml tenant_limits: max_concurrent_conversations must be >= 1")
    return TenantLimits(cap)


def _parties(raw: object) -> dict[str, PartyConfig]:
    data = _mapping(raw, "agent.yaml: 'parties'")
    out: dict[str, PartyConfig] = {}
    for name, item in data.items():
        where = f"agent.yaml parties.{name}"
        party = _mapping(item if item is not None else {}, where)
        _only(party, _PARTY_KEYS, where)
        confirms = party.get("confirms")
        if confirms is not None and not isinstance(confirms, str):
            raise ConfigError(f"{where}: 'confirms' must be an evidence key")
        out[str(name)] = PartyConfig(str(name), _str(party, "description", where, ""), confirms)
    return out


def parse_agent_config(raw: object) -> AgentConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError("agent.yaml must be a mapping")
    _only(raw, _TOP_KEYS, "agent.yaml")
    where = "agent.yaml"
    default_tier, rules = _router(raw.get("router") or {})
    eager = _number(raw, "eager_threshold", where, AgentConfig.eager_threshold)
    if not 0.0 < eager < 1.0:
        raise ConfigError("agent.yaml: eager_threshold must be between 0 and 1")
    retries = _int(raw, "max_policy_retries", where, AgentConfig.max_policy_retries)
    if retries is None or retries < 0:
        raise ConfigError("agent.yaml: max_policy_retries must be >= 0")
    runtime_tools = raw.get("runtime_tools") or []
    if not isinstance(runtime_tools, list) or not all(isinstance(t, str) for t in runtime_tools):
        raise ConfigError("agent.yaml: runtime_tools must be a list of names")
    for name in runtime_tools:
        if name not in RUNTIME_TOOLS:
            raise ConfigError(
                f"agent.yaml: unknown runtime tool {name!r}; known: {sorted(RUNTIME_TOOLS)}"
            )
    rollback_on = raw.get("rollback_on") or []
    if not isinstance(rollback_on, list) or not all(isinstance(o, str) for o in rollback_on):
        raise ConfigError("agent.yaml: rollback_on must be a list of outcome names")
    bridges = raw.get("bridge_phrases") or []
    if not isinstance(bridges, list) or not all(isinstance(p, str) and p for p in bridges):
        raise ConfigError("agent.yaml: bridge_phrases must be a list of phrases")
    bridge_after = _int(raw, "bridge_after_ms", where, AgentConfig.bridge_after_ms)
    if bridge_after is None or bridge_after < 0:
        raise ConfigError("agent.yaml: bridge_after_ms must be >= 0")
    parties = _parties(raw.get("parties") or {})
    if parties and "call_party" not in runtime_tools:
        raise ConfigError("agent.yaml: 'parties' needs runtime tool 'call_party'")
    return AgentConfig(
        name=_str(raw, "name", where),
        description=_str(raw, "description", where, ""),
        system_prompt=_str(raw, "system_prompt", where),
        voice=_voice(raw.get("voice") or {}),
        router_default_tier=default_tier,
        router_rules=rules,
        eager_threshold=eager,
        tenant_limits=_limits(raw.get("tenant_limits") or {}),
        max_policy_retries=retries,
        handoff_message=_str(raw, "handoff_message", where, DEFAULT_HANDOFF_MESSAGE),
        runtime_tools=tuple(dict.fromkeys(runtime_tools)),
        parties=MappingProxyType(parties),
        rollback_on=tuple(dict.fromkeys(rollback_on)),
        bridge_phrases=tuple(bridges),
        bridge_after_ms=bridge_after,
    )
