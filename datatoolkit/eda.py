from abc import ABC, abstractmethod
from typing import Sequence, Union

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy
from typeguard import typechecked


class Group:
    """
    A class for creating groups based on a feature and grouping variables.

    Parameters:
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
    def __init__(self, feature: str, by: list[int | str], data: pd.DataFrame):
        """
        Initialize the Group class with the specified feature, by group(s), and data.

        Args:
            feature (str): The name of the feature to analyze.
            by (list[int | str]): The column(s) to group the data by.
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

    def get_count(self):
        """
        Returns a pandas DataFrame with the count of occurrences of each unique value in the specified feature column.
        Also adds a cumulative count column to the DataFrame.
        """
        self.summarized_data = self.grouped.count.to_frame(name=f"count_{self.feature}")

        self.summarized_data[f"cum_count_{self.feature}"] = self.summarized_data[
            f"count_{self.feature}"
        ].cumsum()

    def get_proportion(self):
        """
        Calculates the proportion and cumulative proportion of the feature in the dataset.
        The results are stored in the summarized_data attribute of the Group object.
        """
        self.summarized_data[f"proportions_{self.feature}"] = (
            self.summarized_data[f"count_{self.feature}"]
            / self.summarized_data[f"count_{self.feature}"].sum()
        )

        self.summarized_data[f"cum_proportions_{self.feature}"] = self.summarized_data[
            f"proportions_{self.feature}"
        ].cumsum()

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
    """Aggregates numerical data frame and provides summary"""

    def __init__(
        self,
        feature: str,
        by: list[int | str],
        data: pd.DataFrame,
        bins: Union[Sequence, str, int] = "auto",
        summary_dict: dict = {
            "sum": self.grouped.sum,
            "min": self.grouped.min,
            "mean": self.grouped.mean,
            0.25: self.grouped.quantile,
            0.5: self.grouped.median,
            0.75: self.grouped.quantile,
            "max": self.grouped.max,
        },
    ):
        super().__init__(feature=feature, by=by, data=data)

        self.bins = bins
        self.summary_dict = summary_dict

    def make_bins(self) -> np.ndarray:
        """Create bins for the data

        Returns:
            np.ndarray: Array of bin edges
        """
        return np.histogram_bin_edges(self.data[self.feature].values, bins=self.bins)

    def binarize(self, fun: str = None):
        pass

    def get_statistics(self) -> pd.DataFrame:
        """Calculates summary statistics in each bin

        Returns:
            (pd.DataFrame): Statistics summary
        """
        self.get_summary()

        if np.issubdtype(self.data[self.feature].dtype, np.number):
            for stat, fun in self.summary_dict.items():
                if isinstance(stat, float):
                    self.summarized_data[f"{stat*100}%_{self.feature}"] = fun(stat)
                else:
                    self.summarized_data[f"{stat}_{self.feature}"] = fun()

        return self.summarized_data

    def __call__(self):
        return self.get_statistics()


class GroupCategorical(Group):
    def summarize(self):
        """Calculates summary statistics in each bin

        Returns:
            (pd.DataFrame): Statistics summary
        """
        grouped = self.data.groupby(self.groupby_args)[self.feature]

        output = grouped.count().to_frame(name=f"count_{self.feature}")
        output[f"cum_count_{self.feature}"] = output[f"count_{self.feature}"].cumsum()
        output[f"proportions_{self.feature}"] = (
            output[f"count_{self.feature}"] / output[f"count_{self.feature}"].sum()
        )
        output[f"cum_proportions_{self.feature}"] = output[
            f"proportions_{self.feature}"
        ].cumsum()

        return output

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.data}, feature={self.feature})"

    def __str__(self):
        return f"{self.__class__.__name__}(data={self.data}, feature={self.feature})"

    def __call__(self, fun: str = None):
        return self.binarize(fun)
