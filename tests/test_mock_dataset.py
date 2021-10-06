from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.resolve()

from datatoolkit.mock_dataset import mock_dataset


@pytest.fixture(scope="module")
def get_data() -> pd.DataFrame:

    specs = {"float": [100, 1, 0.05] \
                    ,"int": [100, 1, 0.025] \
                    ,"categorical": [100, 1, 0.1] \
                    ,"bool": [100, 1, 0] \
                    ,"str": [100, 1, 0] \
                    ,"datetime": [100, 1, 0] \
            }

    data, meta_data = mock_dataset(specs, True)

    return data

def test_shape(get_data):
    data = get_data
    assert data.shape == (100, 6)