"""LLM interface, router, prompt assembly, fallback chain, scripted fake LLM. Imports core only."""

from switchboard.brain.fallback import Attempt, FallbackExhaustedError, FallbackLLM, ServeRecord
from switchboard.brain.llm import LLM, Reply, ScriptedLLM, ScriptExhaustedError, ScriptStep
from switchboard.brain.prompt import (
    AssembledPrompt,
    VolatileContext,
    assemble_prompt,
    canonical_json,
)
from switchboard.brain.router import (
    DEFAULT_ROUTER_CONFIG,
    HeuristicRouter,
    RouteDecision,
    RouteExplanation,
    Router,
    RouterConfig,
    RouteRule,
    RouteState,
    RuleCheck,
)
from switchboard.brain.types import (
    BrainChunk,
    BrainRequest,
    Done,
    Message,
    Role,
    TextDelta,
    Tier,
    ToolCallChunk,
    ToolSchema,
    Usage,
)

__all__ = [
    "DEFAULT_ROUTER_CONFIG",
    "LLM",
    "AssembledPrompt",
    "Attempt",
    "BrainChunk",
    "BrainRequest",
    "Done",
    "FallbackExhaustedError",
    "FallbackLLM",
    "HeuristicRouter",
    "Message",
    "Reply",
    "Role",
    "RouteDecision",
    "RouteExplanation",
    "RouteRule",
    "RouteState",
    "Router",
    "RouterConfig",
    "RuleCheck",
    "ScriptExhaustedError",
    "ScriptStep",
    "ScriptedLLM",
    "ServeRecord",
    "TextDelta",
    "Tier",
    "ToolCallChunk",
    "ToolSchema",
    "Usage",
    "VolatileContext",
    "assemble_prompt",
    "canonical_json",
]
