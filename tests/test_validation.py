
import numpy as np
from models.autoregressive.validation import validate_ar


def test_ar2_seed42():
    passed = validate_ar([0.4, 0.2], seed=42)

    assert np.array_equal(
        passed, [True, True, False, True]
    )


def test_ar2_seed43():
    passed = validate_ar([0.4, 0.2], seed=43)

    assert np.all(passed)


def test_ar3():
    passed = validate_ar([0.5, 0.2, -0.1], seed=2026)

    assert np.all(passed)
