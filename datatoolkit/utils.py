import subprocess
import sys
from abc import ABC, abstractmethod
from collections import Sequence
from collections.abc import Iterable
from pathlib import Path
from typing import Union
from datetime import datetime, timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from typeguard import typechecked

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
                                stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))

sys.path.append(PROJECT_ROOT)

# from tests.mock_dataset import mock_dataset
# from src.make_logger import log_fun

class Group(ABC):
    @typechecked
    def __init__(self, feature: str, data: pd.DataFrame, secondary_feature: str=None
                ,bins: Union[Sequence, str, int]="auto"):
        """Aggregates data frame and provides summary

        Args:
            feature (str): Feature to be agregated
            data (pd.DataFrame): Data frame containing feature
            secondary_feature (str, optional): Secondary feature to be agregated. Defaults to None.
            bins (Union[Sequence, str, int], optional): Bins used to quantize the data. Defaults to "auto".
        """
        self.feature = feature
        self.data = data
        self.bins = bins
        self.secondary_feature = secondary_feature or feature


    def make_bins(self):
        return np.histogram_bin_edges(self.data[self.feature].values
                                    ,bins=self.bins)


    @abstractmethod
    def binarize(self, fun: str=None):
        pass


    def summarize(self):
        """Calculates summary statistics in each bin

        Returns:
            (pd.DataFrame): Statistics summary
        """
        grouped = self.data.groupby(self.groupby_args)[self.secondary_feature]

        summary_dict = {"count": grouped.count
            ,"sum": grouped.sum
            ,"min": grouped.min
            ,"mean": grouped.mean
            ,"25%": grouped.quantile
            ,"50%": grouped.median
            ,"75%": grouped.quantile
            ,"max": grouped.max
            }

        output = summary_dict["count"]().to_frame(name=f"count_{self.secondary_feature}")
        output[f"cum_count_{self.secondary_feature}"] = output[f"count_{self.secondary_feature}"].cumsum()
        output[f"proportions_{self.secondary_feature}"] = output[f"count_{self.secondary_feature}"]/output[f"count_{self.secondary_feature}"].sum()
        output[f"cum_proportions_{self.secondary_feature}"] = output[f"proportions_{self.secondary_feature}"].cumsum()

        if np.issubdtype(self.data[self.secondary_feature].dtype, np.number):
            output[f"min_{self.secondary_feature}"] = summary_dict["min"]()
            output[f"mean_{self.secondary_feature}"] = summary_dict["mean"]()
            output[f"25%_{self.secondary_feature}"] = summary_dict["25%"](0.25)
            output[f"50%_{self.secondary_feature}"] = summary_dict["50%"]()
            output[f"75%_{self.secondary_feature}"] = summary_dict["25%"](0.75)
            output[f"max_{self.secondary_feature}"] = summary_dict["max"]()

        return output
        

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.data}, feature={self.feature})"


    def __str__(self):
        return f"{self.__class__.__name__}(data={self.data}, feature={self.feature})"


    def __call__(self, fun: str=None):
        return self.binarize(fun)


class Quantize(Group):
    """Quantize data frame

    Example:
        >>> data = pd.DataFrame(np.random.rand(10), columns=["A"])
        >>> quantized_data = Quantize(data=data, feature="A")
        >>> _ = quantized_data()
        >>> _ = quantized_data.summarize()
    """
    def binarize(self, fun: str=None):
        if isinstance(self.bins, (str, int, Sequence)):
            self.bins = self.make_bins()

        self.groupby_args = pd.cut(self.data[self.feature].values
                                ,bins=self.bins)

        output = pd.DataFrame(self.groupby_args, columns=["intervals"])
        output["quantized"] = output["intervals"].apply(lambda x: x.mid)
        output[self.feature] = self.data[self.feature].values

        return output


class QuantizeDatetime(Group):
    """Quantize datetime data frame

    Example:
        >>> data = pd.DataFrame(np.arange(datetime(1985,7,1), datetime(2015,7,1), timedelta(days=1)).astype(datetime), columns=["A"])
        >>> quantized_data = QuantizeDatetime(data=data, feature="A", bins="M")
        >>> _ = quantized_data("count")
        >>> _ = quantized_data.summarize()
    """
    bin_time_freq = ["D", "W", "M", "Q", "Y"]

    def binarize(self, fun: str=None):
        if self.bins not in self.bin_time_freq:
            msg = f"{self.bins} is not a valid bin time frequency."
            raise ValueError(msg)

        self.groupby_args = pd.Grouper(key=self.feature, freq=self.bins)

        return getattr(self.data.groupby(self.groupby_args)[self.secondary_feature], fun)()


# @log_fun
@typechecked
def make_pivot(feature: str, index: str, column: str, data: pd.DataFrame
                ,groupby_args: list=None):
    """Create two types of pivot matrices: count and mean

    Args:
        feature (str): Feature that is used as a value for the pivot tables. Needs to be numeric
        index (str): Name of rows of the pivot table
        column (str): Name of columns of the pivot table
        data (pd.DataFrame): Data frame containing the data
        groupby_args (list, optional): Parse arguments to groupby. Defaults to None.

    Returns:
        (pd.DataFrame): Pivot tables
    """


    groupby_args = groupby_args or [index, column]

    grouped = data.groupby(groupby_args)[feature].count().to_frame(name=f"count_{feature}")

    try:
        grouped[f"mean_{feature}"] = data.groupby(groupby_args)[feature].mean()

    except ValueError:
        if np.issubdtype(data[feature].dtype, np.number):
            msg = f"Expected feature {feature} to of data type numerical. Got {data[feature].dtype}."
            raise(msg)

        raise

    grouped.reset_index(inplace=True)
    grouped.sort_values(by=[index, column], inplace=True, ascending=False)

    pivot_count = pd.pivot(grouped, index=index, columns=column, values=f"count_{feature}")
    pivot_mean = pd.pivot(grouped, index=index, columns=column, values=f"mean_{feature}")

    pivot_count.sort_index(inplace=True, ascending=False)
    pivot_mean.sort_index(inplace=True, ascending=False)

    return pivot_count, pivot_mean


# @log_fun
@typechecked
def get_high_frequency_categories(array: Iterable, top_pct_obs: float=0.8
                                ,top_pct_cat: float=0.2):
    """Truncates data according to the proportion of a categorical column

    Args:
        array (Iterable): 1d array containing categories
        top_pct_obs (float): Top percent observations. Defaults to 0.8
        top_pct_cat (float): Top percent categories. Defaults to 0.2

    Returns:
        (Iterable, pd.DataFrame): 1d array with most frequent categories and 
                                    summary statistics

    References:
        [1] https://hsteinshiromoto.github.io/posts/2020/06/25/find_row_closest_value_to_input

    Example:
        >>> s = (np.random.pareto(3, 1000) + 1) * 2
        >>> output, stats = get_high_frequency_categories(s)
    """
    unique, counts = np.unique(array, return_counts=True)
    grouped = pd.DataFrame.from_dict({"category": unique
                                    ,"n_observations": counts
                                    })
    grouped.sort_values(by="n_observations", ascending=False, inplace=True)
    grouped["n_observations_proportions"] = grouped["n_observations"] / grouped["n_observations"].sum()
    grouped["cum_n_observations_proportions"] = grouped["n_observations_proportions"].cumsum()
    grouped["cum_n_categories_proportions"] = np.linspace(1.0/float(grouped.shape[0]), 1, grouped.shape[0])
    grouped.reset_index(inplace=True, drop=True)

    if (top_pct_obs > 0) & (top_pct_cat > 0):
        subset = grouped["cum_n_observations_proportions"] + grouped["cum_n_categories_proportions"]
        threshold = top_pct_obs + top_pct_cat

        # Get row containing values closed to a value [1]
        idx = subset.sub(threshold).abs().idxmin()

    elif (top_pct_obs > 0):
        idx = grouped["cum_n_observations_proportions"].sub(top_pct_obs).abs().idxmin()

    elif (top_pct_cat > 0):
        idx = grouped["cum_n_categories_proportions"].sub(top_pct_cat).abs().idxmin()

    grouped.loc[idx+1, "category"] = "other categories"
    grouped.loc[idx+1, "cum_n_observations_proportions"] = 1

    grouped.loc[idx+1, "n_observations"] = grouped.loc[idx:, "n_observations"].sum()
    grouped.loc[idx+1, "n_observations_proportions"] = grouped.loc[idx:, "n_observations_proportions"].sum()

    return grouped.loc[:idx+1, "category"].values, grouped.loc[:idx+1, :]


def make_graph(nodes: Iterable, M: np.ndarray, G: nx.classes.digraph.DiGraph=nx.DiGraph()):
    """Build graph based on list of nodes and a weight matrix
    Args:
        nodes (list): Graph nodes
        M (np.ndarray): Weight matrix
        G (nx.classes.digraph.DiGraph, optional): Graph type. Defaults to nx.DiGraph().
    Returns:
        [type]: Graph object
    Example:
        >>> n_nodes = 4
        >>> M = np.random.rand(n_nodes, n_nodes)
        >>> nodes = range(M.shape[0])
        >>> G = make_graph(nodes, M)
    """

    for node in nodes:
        G.add_node(node, label=f"{node}")
        
    for i, origin_node in enumerate(nodes):
        for j, destination_node in enumerate(nodes):
            if M[i, j] != 0:
                G.add_edge(origin_node, destination_node, weight=M[i, j]
                            ,label=f"{M[i, j]:0.02f}")

    return G
