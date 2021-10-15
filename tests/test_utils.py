import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[1].resolve()

sys.path.append(str(PROJECT_ROOT))

from datatoolkit import Quantize, QuantizeDatetime, mock_dataset


@pytest.fixture(scope="module")
def get_data() -> pd.DataFrame:

    specs = {"float": [100, 1, 0]
            ,"int": [100, 1, 0]
            ,"categorical": [100, 1, 0]
            ,"bool": [100, 1, 0]
            ,"str": [100, 1, 0]
            ,"datetime": [100, 1, 0]
            }

    data, meta_data = mock_dataset(specs, True)

    return data



def test_Quantize_single_feature_float(get_data):
    data = get_data
    
    quantized_data = Quantize(data=data, feature="float_0")
    qdata = quantized_data()
    sdata = quantized_data.summarize()

    assert qdata.shape == data.shape