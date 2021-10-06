import sys
from datatoolkit.mock_dataset import mock_dataset
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.resolve()

sys.path.append(str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def get_data() -> pd.DataFrame:

    specs = {"float": [100, 1, 0.05], "int": [100, 1, 0.025], "categorical": [100, 1, 0.1], "bool": [100, 1, 0], "str": [100, 1, 0], "datetime": [100, 1, 0]
             }

    data, meta_data = mock_dataset(specs, True)

    return data


def test_shape(get_data):
    data = get_data
    assert data.shape == (100, 6)


def test_nulls_proportion(get_data):
    data = get_data
    assert data.isnull().sum()["float_0"] / data.shape[0] == 0.05
    assert data.isnull().sum()["int_0"] / data.shape[0] == 0.02
    assert data.isnull().sum()["categorical_0"] / data.shape[0] == 0.1
    assert data.isnull().sum()["bool_0"] / data.shape[0] == 0
    assert data.isnull().sum()["str_0"] / data.shape[0] == 0
    assert data.isnull().sum()["datetime_0"] / data.shape[0] == 0
