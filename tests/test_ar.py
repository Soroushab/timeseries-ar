import pytest
from models.autoregressive.simulation import simulate_ar


def test_simulate_ar3():
    result = simulate_ar(
        phi=[0.5, 0.2, -0.1],
        initial_values=[4, 8, 10],
        shocks=[1, -2]
    )

    assert result == pytest.approx([4, 8, 10, 7.2, 2.8])