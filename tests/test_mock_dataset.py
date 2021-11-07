import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[1].resolve()

sys.path.append(str(PROJECT_ROOT))

from datatoolkit import MockData, DataTypes

@pytest.fixture(scope="module")
def get_data() -> pd.DataFrame:

    specs = {DataTypes.float: [100, 1, 0.05]
            ,DataTypes.int: [100, 1, 0.025]
            ,DataTypes.cat: [100, 1, 0.1]
            ,DataTypes.bool: [100, 1, 0]
            ,DataTypes.str: [100, 1, 0]
            ,DataTypes.dt: [100, 1, 0]
            }

    md = MockData(specs)
    data = md()
    meta_data = md.make_meta_data()

    return data


def test_shape(get_data):
    data = get_data
    assert data.shape == (100, 6)


def test_nulls_proportion(get_data):
    data = get_data
    assert data.isnull().sum()["float_0"] / data.shape[0] == 0.05
    assert data.isnull().sum()["int_0"] / data.shape[0] == 0.02
    assert data.isnull().sum()["cat_0"] / data.shape[0] == 0.1
    assert data.isnull().sum()["bool_0"] / data.shape[0] == 0
    assert data.isnull().sum()["str_0"] / data.shape[0] == 0
    assert data.isnull().sum()["dt_0"] / data.shape[0] == 0
