import pandas as pd
import pytest

from src.analysis import (
    DURATION_COLUMN,
    ENERGY_COLUMN,
    classify_probability,
    clean_data,
    train_linear_regression,
)


def test_classify_probability_boundaries():
    assert classify_probability(0.05) == "rare"
    assert classify_probability(0.20) == "unlikely"
    assert classify_probability(0.50) == "likely"
    assert classify_probability(0.95) == "almost certain"


def test_clean_data_removes_invalid_rows_and_duplicates():
    data = pd.DataFrame(
        {
            DURATION_COLUMN: [1, 2, "invalid", 2],
            ENERGY_COLUMN: [10, 20, 30, 20],
        }
    )

    clean = clean_data(data)

    assert len(clean) == 2
    assert clean[DURATION_COLUMN].tolist() == [1.0, 2.0]
    assert clean[ENERGY_COLUMN].tolist() == [10, 20]


def test_regression_requires_minimum_rows():
    data = pd.DataFrame(
        {
            DURATION_COLUMN: [1, 2, 3, 4],
            ENERGY_COLUMN: [10, 20, 30, 40],
        }
    )

    with pytest.raises(ValueError):
        train_linear_regression(data)
