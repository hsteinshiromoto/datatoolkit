import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from scipy.stats import entropy
from typeguard import typechecked

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(f"{PROJECT_ROOT}")

from datatoolkit.utils import flatten


@dataclass(kw_only=True)
class Discretize:
    """
    A class for discretizing a continuous column into bins.

    Attributes:
        make_bin_edges (method): A method that computes the edges of the bins.
        get_labels (method): A method that returns the labels for each bin.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        >>> d = Discretize()
        >>> print(bin_edges := d.make_bin_edges(data=df, col_name='A', bins=2))
        [1. 3. 5.]
        >>> d.get_labels(bin_edges)
        ['1.00-3.00', '3.00-5.00']
    """

    @staticmethod
    def make_bin_edges(
        data: pd.DataFrame, col_name: str, bins: Union[Sequence, str, int]
    ) -> np.ndarray[np.number]:
        """
        Compute the bin edges for the given feature using the specified number of bins.

        Args:
            data (pd.DataFrame): The input data.
            col_name (str): The name of the column to compute bin edges for.
            bins (Union[Sequence, str, int]): The number of bins to use or the bin edges.

        Returns:
            np.ndarray[np.number]: The computed bin edges.
        """
        return np.histogram_bin_edges(data[col_name], bins=bins)

    @staticmethod
    def get_labels(bin_edges: np.ndarray[np.number]) -> list[str]:
        """
        Returns a list of labels for each bin in the histogram.

        Args:
            bin_edges (np.ndarray[np.number]): The bin edges.

        Returns:
            list[str]: The labels for each bin in the histogram.
        """
        return [
            f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}"
            for i in range(len(bin_edges) - 1)
        ]


@dataclass(kw_only=True)
class Group(Discretize):
    """
    A class for creating groups based on a feature and grouping variables.

    Attributes:
    -----------
    by : str | Iterable[int | str | pd.api.types.CategoricalDtype]
        The column(s) to group the data by.

    Methods:
    --------
    __post_init__():
        Initialize the Group class with the specified feature, by group(s), and data.
    make_discretized_groupby_args(by: str, data: pd.DataFrame, bin_edges: Iterable[np.number]):
        Creates a dictionary of arguments to pass to the groupby function for discretized data.
    make_datetime_groupby_args(by: Union[str, Iterable[Union[int, str]]], bins: str):
        Creates a dictionary of arguments to pass to the groupby function for datetime data.
    make_groups(groupby_args):
        Creates groups based on the feature.

    Example:
    --------
    >>> df = pd.DataFrame({
    ... "date": pd.date_range('2015-02-24', periods=8, freq='D'),
    ... 'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
    ... 'age': [25, 30, 35, 40, 45, 50, 55, 60],
    ... 'income': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000]
    ... })
    >>> group = Group(feature='income', by=['gender', 'date'], data=df, bins='W')
    >>> group.grouped.sum() # doctest: +NORMALIZE_WHITESPACE
    gender  date
    F       2015-03-01    240000
            2015-03-08    120000
    M       2015-03-01    210000
            2015-03-08    110000
    Name: income, dtype: int64
    >>> group = Group(feature='income', by=['gender'], data=df)
    >>> group.grouped.sum() # doctest: +NORMALIZE_WHITESPACE
    gender
    F    360000
    M    320000
    Name: income, dtype: int64
    >>> group = Group(feature='income', by=['age'], data=df, bins='auto')
    >>> group.grouped.sum() # doctest: +NORMALIZE_WHITESPACE
    (25.0, 33.75]     60000
    (33.75, 42.5]    150000
    (42.5, 51.25]    190000
    (51.25, 60.0]    230000
    Name: income, dtype: int64
    """

    feature: str
    by: Union[str, Iterable[Union[int, str, pd.api.types.CategoricalDtype]]]
    data: pd.DataFrame
    bins: Union[Sequence, str, int] = None

    @typechecked
    def __post_init__(self):
        """
        Initialize the Group class with the specified feature, by group(s), and data.

        Args:
            feature (str): The name of the feature to analyze.
            by (list[Union[int, str]]): The column(s) to group the data by.
            data (pd.DataFrame): The input data to analyze.
        """
        if self.bins in ["D", "W", "M", "Q", "Y"]:
            datetime_col = [col for col in self.by if is_datetime(self.data[col])]
            self.by = [col for col in self.by if col not in datetime_col]

            self.data.set_index(datetime_col, inplace=True)

            groupby_args = self.make_datetime_groupby_args(self.by, self.bins)

        elif self.bins:
            bin_edges = self.make_bin_edges(
                data=self.data, col_name=self.by, bins=self.bins
            )
            groupby_args = self.make_discretized_groupby_args(
                self.by, self.data, bin_edges
            )

        else:
            groupby_args = self.by

        self.make_groups(groupby_args)

    @staticmethod
    def make_discretized_groupby_args(
        by: str,
        data: pd.DataFrame,
        bin_edges: Iterable[np.number],
    ):
        """
        Creates a dictionary of arguments to pass to the groupby function for discretized data.

        Args:
            by (str): The column to group the data by.
            data (pd.DataFrame): The input data to analyze.
            bin_edges (Iterable[np.number]): The bin edges to use for discretization.

        Returns:
            A dictionary of arguments to pass to the groupby function.
        """
        flattened = np.array(list(flatten(data[by].values)))

        return pd.cut(flattened, bins=bin_edges)

    @staticmethod
    def make_datetime_groupby_args(
        by: Union[str, Iterable[Union[int, str]]],
        bins: str,
    ):
        """
        Creates a dictionary of arguments to pass to the groupby function for datetime data.

        Args:
            by (Union[str, Iterable[Union[int, str]]]): The column(s) to group the data by.
            bins (str): The frequency to group the data by.

        Returns:
            A dictionary of arguments to pass to the groupby function.
        """
        if isinstance(by, Iterable):
            by.extend([pd.Grouper(freq=bins)])

            return by

        elif isinstance(by, str):
            [by].extend([pd.Grouper(freq=bins)])
            return by

        else:
            raise TypeError(
                f"Expected by to be str or Iterable[Union[int, str]], got {type(by)}"
            )

    def make_groups(self, groupby_args):
        """
        Creates groups based on the feature.

        Args:
            groupby_args: The arguments to pass to the groupby function.
        """
        self.grouped = self.data.groupby(groupby_args)[self.feature]

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.data}, features={self.by})"

    def __str__(self):
        return f"{self.__class__.__name__}(data={self.data}, features={self.by})"


@dataclass(kw_only=True)
class Summarize(Group):
    """
    A class used to summarize data by groups.

    Attributes
    ----------
    feature : str
        The name of the feature to be summarized.
    data : pandas.DataFrame
        The data to be summarized.
    groups : list
        The list of groups to be used for summarization.

    Methods
    -------
    get_count()
        Calculates the count of each group.
    get_proportion()
        Calculates the proportion of each group.
    get_entropy()
        Calculates the entropy of each group.
    make_summary()
        Summarizes the data by groups.

    Example
    -------
    >>> data = pd.DataFrame({'by': ['A', 'A', 'B', 'B', 'B', 'C'], 'feature': [1, 2, 3, 1, 2, 3]})
    >>> summarize = Summarize(feature='feature', by=['by'], data=data)
    >>> summarized_data = summarize.make_summary()
    >>> summarized_data[['count_feature', 'cum_count_feature']]  # doctest: +NORMALIZE_WHITESPACE
        count_feature  cum_count_feature
    by
    A               2                  2
    B               3                  5
    C               1                  6
    >>> summarized_data[['proportions_feature', 'cum_proportions_feature']]  # doctest: +NORMALIZE_WHITESPACE
        proportions_feature  cum_proportions_feature
    by
    A              0.333333                 0.333333
    B              0.500000                 0.833333
    C              0.166667                 1.000000
    >>> summarized_data[['entropy_feature']]  # doctest: +NORMALIZE_WHITESPACE
            entropy_feature
    by
    A          0.636514
    B          1.011404
    C          0.000000
    """

    def get_count(self, summarized_data: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a pandas DataFrame with the count of occurrences of each unique value in the specified feature column.
        Also adds a cumulative count column to the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with the count of occurrences of each unique value in the specified feature column.
        """
        summarized_data[f"count_{self.feature}"] = self.grouped.count()

        summarized_data[f"cum_count_{self.feature}"] = summarized_data[
            f"count_{self.feature}"
        ].cumsum()

        return summarized_data

    def get_proportion(self, summarized_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the proportion and cumulative proportion of the feature in the dataset.
        The results are stored in the summarized_data attribute of the Group object.

        Returns:
            pd.DataFrame: A DataFrame with the proportion and cumulative proportion of the feature in the dataset.
        """
        summarized_data[f"proportions_{self.feature}"] = (
            summarized_data[f"count_{self.feature}"]
            / summarized_data[f"count_{self.feature}"].sum()
        )

        summarized_data[f"cum_proportions_{self.feature}"] = summarized_data[
            f"proportions_{self.feature}"
        ].cumsum()

        return summarized_data

    def get_entropy(self, summarized_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the entropy of the grouped data for the specified feature and adds it to the summarized data.

        Returns:
            pd.DataFrame: A DataFrame with the entropy of the feature for each group.
        """
        summarized_data[f"entropy_{self.feature}"] = self.grouped.apply(entropy)

        return summarized_data

    def make_summary(self):
        """
        Generates a summary of the data, including group counts, proportions, and entropy.
        """
        summarized_data = pd.DataFrame()
        summarized_data = self.get_count(summarized_data)
        summarized_data = self.get_proportion(summarized_data)
        summarized_data = self.get_entropy(summarized_data)

        return summarized_data


@dataclass(kw_only=True)
class Numerical(Summarize):
    """
    A class for calculating summary statistics for numerical data.

    Attributes
    ----------
    summary_dict: dict, optional
        A dictionary of summary statistics to calculate. Defaults to {}.

    Methods
    -------
    get_statistics()
        Calculates summary statistics for the data.

    make_bins()
        Create bins for the data

    binarize()
        Binarize the data

    Example
    -------
        >>> data = pd.DataFrame({'by': ['A', 'A', 'B', 'B', 'B', 'C'], 'feature': [1, 2, 3, 1, 2, 3]})
        >>> numerical = Numerical(feature='feature', by=['by'], data=data)
        >>> numerical.make_num_summary()['mean_feature'] # doctest: +NORMALIZE_WHITESPACE
        by
        A    1.5
        B    2.0
        C    3.0
        Name: mean_feature, dtype: float64
    """

    summary_dict: Optional[Dict[str, Callable]] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        self.summary_dict = self.summary_dict or {
            "sum": self.grouped.sum,
            "min": self.grouped.min,
            "mean": self.grouped.mean,
            0.25: self.grouped.quantile,
            0.5: self.grouped.median,
            0.75: self.grouped.quantile,
            "max": self.grouped.max,
        }

    def get_stats(self, summarized_data: pd.DataFrame) -> pd.DataFrame:
        """Calculates summary statistics for the data

        Raises:
            TypeError: If the feature is not numeric.

        Returns:
            pd.DataFrame: Summary statistics
        """

        if not np.issubdtype(self.data[self.feature].dtype, np.number):
            raise TypeError(f"Expected feature {self.feature} to be numeric")

        for stat, fun in self.summary_dict.items():
            if isinstance(stat, float):
                summarized_data[f"{stat}_{self.feature}"] = fun(stat)
            else:
                summarized_data[f"{stat}_{self.feature}"] = fun()

        return summarized_data

    def make_num_summary(self):
        """Makes numerical summary"""
        summarized_data = self.make_summary()
        summarized_data = self.get_stats(summarized_data)

        return summarized_data
