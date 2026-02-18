import pytest

pytest.importorskip("joblib")
pytest.importorskip("pandas")

from predictor import implied_prob, kelly_fraction


def test_implied_prob():
    assert round(implied_prob(2.0), 6) == 0.5
    assert implied_prob(1.0) == 0.5


def test_kelly_fraction_basic():
    f = kelly_fraction(0.57, 1.85)
    assert f > 0
    assert round(f, 4) == round((0.57 * 0.85 - 0.43) / 0.85, 4)


def test_kelly_fraction_no_value():
    assert kelly_fraction(0.40, 1.85) == 0.0
