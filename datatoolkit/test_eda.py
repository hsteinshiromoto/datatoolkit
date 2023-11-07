import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).parents[1].resolve()

sys.path.append(str(PROJECT_ROOT))

from datatoolkit import Group, mock_dataset, Summarize


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


def test_Group_make_groups(get_data):
    data = get_data
    
    group = Group(feature="float_0", by=["categorical_0"], data=data)
    group.make_groups()

    assert isinstance(group.grouped, pd.core.groupby.SeriesGroupBy)import sys


class TestSummarize:
    def test_get_count(self, get_data):
        data = get_data
        summarize = Summarize(feature="float_0", by=["categorical_0"], data=data)
        summarize.get_count()

        assert isinstance(summarize.summarized_data, pd.DataFrame)
        assert "count_float_0" in summarize.summarized_data.columns
        assert "cum_count_float_0" in summarize.summarized_data.columns

    def test_get_proportion(self, get_data):
        data = get_data
        summarize = Summarize(feature="float_0", by=["categorical_0"], data=data)
        summarize.get_proportion()

        assert isinstance(summarize.summarized_data, pd.DataFrame)
        assert "proportions_float_0" in summarize.summarized_data.columns
        assert "cum_proportions_float_0" in summarize.summarized_data.columns

    def test_get_entropy(self, get_data):
        data = get_data
        summarize = Summarize(feature="float_0", by=["categorical_0"], data=data)
        summarize.get_entropy()

        assert isinstance(summarize.summarized_data, pd.DataFrame)
        assert "entropy_float_0" in summarize.summarized_data.columns

    def test_get_summary(self, get_data):
        data = get_data
        summarize = Summarize(feature="float_0", by=["categorical_0"], data=data)
        summarize.get_summary()

        assert isinstance(summarize.summarized_data, pd.DataFrame)
        assert "count_float_0" in summarize.summarized_data.columns
        assert "cum_count_float_0" in summarize.summarized_data.columns
        assert "proportions_float_0" in summarize.summarized_data.columns
        assert "cum_proportions_float_0" in summarize.summarized_data.columns
        assert "entropy_float_0" in summarize.summarized_data.columns