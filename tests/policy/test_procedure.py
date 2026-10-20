import pytest

from switchboard.policy.procedure import ProcedureGraph, ProcedureState
from tests.policy.helpers import advanced, executed


def graph() -> ProcedureGraph:
    return ProcedureGraph(
        steps=("greet", "verify_identity", "resolve", "close"),
        require_before={
            "lookup_order": ("verify_identity",),
            "issue_refund": ("resolve", "verify_identity"),
        },
        completes={"verify_identity": "verify_identity", "check_eligibility": "resolve"},
    )


def test_open_steps_in_declared_order() -> None:
    g = graph()
    assert g.open_steps(ProcedureState.empty()) == ("greet", "verify_identity", "resolve", "close")
    state = ProcedureState(frozenset({"verify_identity", "greet"}))
    assert g.open_steps(state) == ("resolve", "close")


def test_procedure_advanced_completes_step() -> None:
    state = graph().state_from_events([advanced("greet"), advanced("verify_identity", seq=2)])
    assert state.completed == frozenset({"greet", "verify_identity"})


def test_successful_tool_execution_completes_mapped_step() -> None:
    state = graph().state_from_events([executed("verify_identity", result={"ok": True})])
    assert state.completed == frozenset({"verify_identity"})
    state = graph().state_from_events([executed("check_eligibility", result={"eligible": True})])
    assert state.completed == frozenset({"resolve"})


@pytest.mark.parametrize("result", [{"ok": False}, {"error": "timeout"}])
def test_failed_tool_execution_does_not_complete_step(result: dict[str, object]) -> None:
    state = graph().state_from_events([executed("verify_identity", result=result)])
    assert state.completed == frozenset()


def test_unmapped_tool_and_unknown_step_do_not_complete() -> None:
    state = graph().state_from_events([executed("lookup_order"), advanced("not_a_step")])
    assert state.completed == frozenset()


def test_missing_prerequisites_in_step_order() -> None:
    g = graph()
    empty = ProcedureState.empty()
    assert g.missing_prerequisites("lookup_order", empty) == ("verify_identity",)
    assert g.missing_prerequisites("issue_refund", empty) == ("verify_identity", "resolve")
    assert g.missing_prerequisites("unconstrained_tool", empty) == ()
    partial = ProcedureState(frozenset({"verify_identity"}))
    assert g.missing_prerequisites("lookup_order", partial) == ()
    assert g.missing_prerequisites("issue_refund", partial) == ("resolve",)


def test_ordering_is_not_transitive() -> None:
    # Steps order the open-step list; only require_before gates tools.
    g = graph()
    state = ProcedureState(frozenset({"resolve", "verify_identity"}))
    assert g.missing_prerequisites("issue_refund", state) == ()
    assert g.open_steps(state) == ("greet", "close")


def test_next_step() -> None:
    g = graph()
    assert g.next_step(ProcedureState.empty()) == "greet"
    assert g.next_step(ProcedureState(frozenset(g.steps))) is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"steps": ("a", "a")}, "duplicate step 'a'"),
        ({"steps": ("a",), "require_before": {"t": ("b",)}}, "unknown step 'b'"),
        ({"steps": ("a",), "completes": {"t": "b"}}, "unknown step 'b'"),
    ],
)
def test_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ProcedureGraph(**kwargs)  # type: ignore[arg-type]
