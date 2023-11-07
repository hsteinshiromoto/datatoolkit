from collections.abc import Sequence
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import entropy
from typeguard import typechecked


class Group:
    """
    A class for creating groups based on a feature and grouping variables.

    Attributes:
    -----------
    feature : str
        The feature to group by.
    by : list[int | str]
        The variables to group by.
    data : pd.DataFrame
        The data to group.

    Methods:
    --------
    make_groups():
        Creates groups based on the feature.
    """

    @typechecked
    def __init__(self, feature: str, by: list[Union[str, int]], data: pd.DataFrame):
        """
        Initialize the Group class with the specified feature, by group(s), and data.

        Args:
            feature (str): The name of the feature to analyze.
            by (list[Union[int, str]]): The column(s) to group the data by.
            data (pd.DataFrame): The input data to analyze.
        """
        self.by = by
        self.data = data
        self.feature = feature or by

    def make_groups(self):
        """Creates groups based on the feature"""
        self.grouped = self.data.groupby(self.by)[self.feature]

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.data}, features={self.by})"

    def __str__(self):
        return f"{self.__class__.__name__}(data={self.data}, features={self.by})"


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
    get_summary()
        Summarizes the data by groups.
    """

    def __init__(self, feature: str, by: list[Union[int, str]], data: pd.DataFrame):
        super().__init__(feature=feature, by=by, data=data)

        self.make_groups()
        self.get_count()

    def get_count(self):
        """
        Returns a pandas DataFrame with the count of occurrences of each unique value in the specified feature column.
        Also adds a cumulative count column to the DataFrame.

        Example:
            >>> data = pd.DataFrame({'by': ['A', 'A', 'B', 'B', 'B', 'C'], 'feature': [1, 2, 3, 1, 2, 3]})
            >>> summarize = Summarize(feature='feature', by=['by'], data=data)
            >>> summarize.get_count()  # doctest: +NORMALIZE_WHITESPACE
                count_feature  cum_count_feature
            by
            A               2                  2
            B               3                  5
            C               1                  6
        """
        self.summarized_data = self.grouped.count().to_frame(
            name=f"count_{self.feature}"
        )

        self.summarized_data[f"cum_count_{self.feature}"] = self.summarized_data[
            f"count_{self.feature}"
        ].cumsum()

        return self.summarized_data

    def get_proportion(self):
        """
        Calculates the proportion and cumulative proportion of the feature in the dataset.
        The results are stored in the summarized_data attribute of the Group object.

        Example:
            >>> data = pd.DataFrame({'by': ['A', 'A', 'B', 'B', 'B', 'C'], 'feature': [1, 2, 3, 1, 2, 3]})
            >>> summarize = Summarize(feature='feature', by=['by'], data=data)
            >>> summarize.get_proportion() # doctest: +NORMALIZE_WHITESPACE
                proportions_feature  cum_proportions_feature
            by
            A              0.333333                 0.333333
            B              0.500000                 0.833333
            C              0.166667                 1.000000
        """
        self.summarized_data[f"proportions_{self.feature}"] = (
            self.summarized_data[f"count_{self.feature}"]
            / self.summarized_data[f"count_{self.feature}"].sum()
        )

        self.summarized_data[f"cum_proportions_{self.feature}"] = self.summarized_data[
            f"proportions_{self.feature}"
        ].cumsum()

        return self.summarized_data[
            [f"proportions_{self.feature}", f"cum_proportions_{self.feature}"]
        ]

    def get_entropy(self):
        """
        Calculates the entropy of the grouped data for the specified feature and adds it to the summarized data.

        """
        self.summarized_data[f"entropy_{self.feature}"] = self.grouped.apply(entropy)

    def get_summary(self):
        """
        Generates a summary of the data, including group counts, proportions, and entropy.
        """
        self.make_groups()
        self.get_count()
        self.get_proportion()
        self.get_entropy()


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
    """

    def __init__(
        self,
        feature: str,
        by: list[Union[int, str]],
        data: pd.DataFrame,
        bins: Union[Sequence, str, int] = "auto",
        summary_dict: dict = {},
    ):
        super().__init__(feature=feature, by=by, data=data)

        self.bins = bins
        self.summary_dict = summary_dict or {
            "sum": self.grouped.sum,
            "min": self.grouped.min,
            "mean": self.grouped.mean,
            0.25: self.grouped.quantile,
            0.5: self.grouped.median,
            0.75: self.grouped.quantile,
            "max": self.grouped.max,
        }

    def make_bins(self) -> np.ndarray:
        """Create bins for the data

        Returns:
            np.ndarray: Array of bin edges
        """
        return np.histogram_bin_edges(self.data[self.feature].values, bins=self.bins)

    def binarize(self, fun: str = None):
        pass

    def get_statistics(self) -> pd.DataFrame:
        """Calculates summary statistics for the data

        Raises:
            TypeError: If the feature is not numeric.

        Returns:
            pd.DataFrame: Summary statistics
        """
        self.get_summary()

        if not np.issubdtype(self.data[self.feature].dtype, np.number):
            raise TypeError(f"Expected feature {self.feature} to be numeric")

        for stat, fun in self.summary_dict.items():
            if isinstance(stat, float):
                self.summarized_data[f"{stat*100}%_{self.feature}"] = fun(stat)
            else:
                self.summarized_data[f"{stat}_{self.feature}"] = fun()

        return self.summarized_data

    def __call__(self):
        return self.get_statistics()
