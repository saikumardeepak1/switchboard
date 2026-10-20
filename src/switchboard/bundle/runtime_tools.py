"""Tools the runtime itself provides to bundles that declare them in `agent.yaml`.

Their specs live here, below the runtime, so the bundle compiler can validate policy references
to them and hash them into the bundle id. The runtime supplies the behaviour.
"""

from __future__ import annotations

from types import MappingProxyType

from switchboard.effects import ToolSpec

# Effect "write": calling someone is a real-world action, so it never runs on a speculative turn
# and rules guarding writes apply to it.
CALL_PARTY = ToolSpec(
    name="call_party",
    description=(
        "Place an outbound call to another party while the current caller stays on the line. "
        "Returns at once; the outcome arrives later as a child call result."
    ),
    effect="write",
    params={
        "type": "object",
        "properties": {
            "party": {"type": "string", "description": "Who to call, as named in the bundle."},
            "goal": {"type": "string", "description": "What the call must find out."},
            "questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Questions to ask, in order.",
            },
        },
        "required": ["party", "goal"],
    },
)

RUNTIME_TOOLS: MappingProxyType[str, ToolSpec] = MappingProxyType({CALL_PARTY.name: CALL_PARTY})
