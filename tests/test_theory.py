
import pytest
from models.autoregressive.theory import (
    is_stationary,
    theoretical_autocovariances
)


def test_stationary():
    assert is_stationary([0.4, 0.2])


def test_nonstationary():
    assert not is_stationary([1.2])


def test_theoretical_autocovariances():
    gamma = theoretical_autocovariances([0.4, 0.2])

    assert gamma == pytest.approx([
        1.388888889,
        0.694444444,
        0.555555556
    ])


def test_nonstationary_error():
    with pytest.raises(ValueError):
        theoretical_autocovariances([1.2])
