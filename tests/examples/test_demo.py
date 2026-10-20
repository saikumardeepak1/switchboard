"""The founder-facing demo: runs fast, reads as a transcript, and shows both gates firing."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from examples import demo

ROOT = Path(__file__).resolve().parents[2]
STAMP = re.compile(r"^\s*(\d+):(\d{2}\.\d)\s")


def _seconds(line: str) -> float:
    m = STAMP.match(line)
    assert m is not None, line
    return int(m.group(1)) * 60 + float(m.group(2))


def _parts(out: str) -> tuple[list[str], list[str]]:
    assert "PART 1" in out and "PART 2" in out
    one, two = out.split("PART 2", 1)
    return one.split("PART 1", 1)[1].splitlines(), two.splitlines()


@pytest.fixture(scope="module")
def output() -> str:
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "examples.demo"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert time.perf_counter() - start < 10.0
    return proc.stdout


def test_runs_without_color_when_not_a_tty(output: str) -> None:
    assert "\x1b[" not in output
    assert re.search(r"^\s*\d+:\d{2}\.\d\s+COURIER\s+Hi, this is courier C-77\.", output, re.M)
    assert re.search(r"^\s*\d+:\d{2}\.\d\s+CUSTOMER\s+Yes, I asked the courier", output, re.M)
    assert re.search(r"AGENT -> courier", output)
    assert re.search(r"AGENT -> customer", output)


def test_transcript_is_in_time_order(output: str) -> None:
    for part in _parts(output):
        stamps = [_seconds(line) for line in part if STAMP.match(line)]
        assert stamps and stamps == sorted(stamps)


def test_courier_talks_while_the_customer_leg_is_open(output: str) -> None:
    one, _ = _parts(output)
    timed = [line for line in one if STAMP.match(line)]
    opened = next(i for i, line in enumerate(timed) if "dialing customer" in line)
    result = next(i for i, line in enumerate(timed) if "result" in line and "confirmed" in line)
    assert any("COURIER" in line for line in timed[opened + 1 : result])


def test_part_two_blocks_an_unsafe_action_and_a_wrong_number(output: str) -> None:
    _, two = _parts(output)
    text = "\n".join(two)
    assert re.search(r"BLOCKED mark_delivered: delivered_needs_confirmation", text)
    assert re.search(r'BLOCKED speech ".*\$[\d.]+.*": unsupported \$[\d.]+ \(evidence: \$', text)


def test_mark_delivered_executes_exactly_once_in_each_part(output: str) -> None:
    for part in _parts(output):
        executed = [line for line in part if re.search(r"tool\s+mark_delivered\(.*->", line)]
        assert len(executed) == 1
        assert "key mark_delivered:" in executed[0]


def test_part_two_summary(output: str) -> None:
    _, two = _parts(output)
    text = "\n".join(two)
    assert re.search(r"violating writes\s+proposed [1-9]\d*, executed 0", text)
    assert re.search(r"sentences blocked\s+[1-9]", text)
    assert re.search(r"mark_delivered runs\s+1\b", text)
    assert re.search(r"outcome\s+resolved", text)


def test_footer_points_at_next_steps(output: str) -> None:
    assert "make bench" in output and "make serve" in output and "README" in output


async def test_chosen_seed_produces_both_faults_and_both_blocks() -> None:
    result = await demo.run_call(demo.ADVERSARY, demo.ADVERSARY_SEED)
    kinds = {f.kind for f in result.faults}
    assert {"early_mark_delivered", "corrupt_money"} <= kinds
    blocked = [e for e in result.events if e.type == "policy.verdict" and not e.allowed]
    assert any("delivered_needs_confirmation" in e.rule_ids for e in blocked)
    gated = [e for e in result.events if e.type == "speech.sentence_gated"]
    assert any(e.decision == "block" and "$" in e.sentence for e in gated)
    marks = [e for e in result.events if e.type == "tool.executed" and e.tool == "mark_delivered"]
    assert len(marks) == 1
    assert result.outcome == "resolved"


async def test_clean_run_has_no_blocks() -> None:
    result = await demo.run_call(None, 0)
    assert not [e for e in result.events if e.type == "policy.verdict" and not e.allowed]
    assert result.outcome == "resolved"


def test_color_only_on_request() -> None:
    line = demo.Line(1.0, "BLOCKED mark_delivered: x", "red")
    assert "\x1b[" not in demo.format_line(line, color=False)
    assert "\x1b[31m" in demo.format_line(line, color=True)
