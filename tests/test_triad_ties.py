"""Acceptance tests for tied head-to-head triads."""
import itertools, sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "data" / "scripts"))
from raw_intransitivity import compute_raw_ia, weak_pat


def hist(rows):
    return pd.DataFrame([{"Date": d, "Aname": a, "Bname": b, "Awin": w} for d, a, b, w in rows])


def test_weak_orders_exhaustive():
    sign = lambda x: (x > 0) - (x < 0)
    expected = {(sign(a-b), sign(a-c), sign(b-c)) for a, b, c in itertools.product(range(3), repeat=3)}
    actual = {s for s in itertools.product([-1, 0, 1], repeat=3) if weak_pat(*s)}
    assert actual == expected


CASES = [
    ([("2020-01-01", "A", "B", 1), ("2020-02-01", "A", "B", 0),
      ("2020-03-01", "A", "C", 1), ("2020-04-01", "B", "C", 1)], 1, 0, 1),  # A=B>C
    ([("2020-01-01", "A", "B", 1), ("2020-02-01", "A", "B", 1),
      ("2020-03-01", "A", "C", 1), ("2020-04-01", "B", "C", 1)], 1, 0, 1),
    ([("2020-01-01", "A", "B", 1), ("2020-03-01", "B", "C", 1),
      ("2020-04-01", "C", "A", 1)], 1, 1, 1),
    ([("2020-01-01", "A", "C", 1), ("2020-02-01", "B", "C", 1),
      ("2020-03-01", "A", "D", 1), ("2020-04-01", "B", "D", 0)], 0, 0, 2),
    ([("2020-01-01", "A", "B", 1), ("2020-02-01", "A", "C", 1),
      ("2020-03-01", "A", "C", 0), ("2020-04-01", "B", "C", 1)], 0, 0, 1),
    ([("2020-01-01", "A", "B", 1), ("2020-02-01", "A", "C", 1),
      ("2020-03-01", "B", "C", 1), ("2020-04-01", "B", "C", 0)], 1, 0, 1),  # A>B=C
    ([("2020-01-01", "A", "B", 1), ("2020-02-01", "A", "B", 0),
      ("2020-03-01", "A", "C", 1), ("2020-04-01", "A", "C", 0),
      ("2020-05-01", "B", "C", 1), ("2020-05-02", "B", "C", 0)], 1, 0, 1),  # A=B=C
    ([("2020-01-01", "A", "B", 1), ("2020-02-01", "A", "B", 0),
      ("2020-03-01", "A", "C", 1), ("2020-04-01", "C", "B", 1)], 0, 0, 1),  # inconsistent tie
]


@pytest.mark.parametrize("rows,n_tot,n_int,n_common", CASES)
def test_triad_orientation(rows, n_tot, n_int, n_common):
    query = pd.DataFrame([{"Date": "2020-06-01", "Aname": "A", "Bname": "B",
                           "gender_source": "men"}])
    out = compute_raw_ia(query, hist(rows), min_h2h=1).iloc[0]
    assert (out.n_triads_total, out.n_triads_intransitive, out.n_common_opponents) == (n_tot, n_int, n_common)
    assert out.raw_matchup_confidence > 0
