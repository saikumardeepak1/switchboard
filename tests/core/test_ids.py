import random
import re

from switchboard.core.ids import new_id


def test_new_id_format() -> None:
    assert re.fullmatch(r"conv_[0-9a-f]{8}", new_id("conv", random.Random(1)))


def test_new_id_is_deterministic_per_seed() -> None:
    a = [new_id("leg", random.Random(7)) for _ in range(1)]
    b = [new_id("leg", random.Random(7)) for _ in range(1)]
    assert a == b
    rng = random.Random(7)
    seq = [new_id("leg", rng) for _ in range(3)]
    assert len(set(seq)) == 3
    assert seq[0] == a[0]
