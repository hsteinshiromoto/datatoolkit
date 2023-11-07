import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[1].resolve()

sys.path.append(str(PROJECT_ROOT))

from datatoolkit import DataTypes, Group, MockData, Numerical, Summarize


@pytest.fixture(scope="module")
def get_data() -> pd.DataFrame:
    specs = {
        DataTypes.float: [100, 1, 0],
        DataTypes.int: [100, 1, 0],
        DataTypes.cat: [100, 1, 0],
        DataTypes.bool: [100, 1, 0],
        DataTypes.str: [100, 1, 0],
        DataTypes.dt: [100, 1, 0],
    }

    md = MockData(specs_dict=specs)
    data = md()

    return data


def test_Group_make_groups(get_data):
    data = get_data

    group = Group(feature="float_0", by=["categorical_0"], data=data)
    group.make_groups()

    assert isinstance(group.grouped, pd.core.groupby.SeriesGroupBy)


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


class TestNumerical:
    def test_make_bins(self, get_data):
        data = get_data
        numerical = Numerical(feature="float_0", by=["categorical_0"], data=data)
        bins = numerical.make_bins()

        assert isinstance(bins, np.ndarray)

    def test_get_statistics(self, get_data):
        data = get_data
        numerical = Numerical(feature="float_0", by=["categorical_0"], data=data)
        stats = numerical.get_statistics()

        assert isinstance(stats, pd.DataFrame)
        assert "sum_float_0" in stats.columns
        assert "min_float_0" in stats.columns
        assert "mean_float_0" in stats.columns
        assert "25.0%_float_0" in stats.columns
        assert "50.0%_float_0" in stats.columns
        assert "75.0%_float_0" in stats.columns
        assert "max_float_0" in stats.columns

    def test_get_statistics_with_non_numeric_feature(self, get_data):
        data = get_data
        numerical = Numerical(feature="categorical_0", by=["float_0"], data=data)

        with pytest.raises(TypeError):
            numerical.get_statistics()
