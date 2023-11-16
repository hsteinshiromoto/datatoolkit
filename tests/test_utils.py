import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[1].resolve()

sys.path.append(str(PROJECT_ROOT))

from datatoolkit import mock_dataset


@pytest.fixture(scope="module")
def get_data() -> pd.DataFrame:
    specs = {
        "float": [100, 1, 0],
        "int": [100, 1, 0],
        "categorical": [100, 1, 0],
        "bool": [100, 1, 0],
        "str": [100, 1, 0],
        "datetime": [100, 1, 0],
    }

    data, meta_data = mock_dataset(specs, True)

    return data
