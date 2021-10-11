import subprocess
import sys
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Union

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
    """Group class
    Args:
        name (str): Group name
        members (list): Group members
    """
    @typechecked
    def __init__(self, feature: str, data: pd.DataFrame, secondary_feature: str=None
                ,bins: Union[np.array, str, bool]="auto"):
        self.feature = feature
        self.data = data
        self.bins = bins
        self.secondary_feature = secondary_feature or feature


    def make_bins(self):
        self.bins = self.bins or \
                    np.histogram_bin_edges(self.data[self.feature].values
                                        ,bins=self.bins)


    @abstractmethod
    def binarize(self):
        pass


    def summarize(self):
        groupby_args = self.group_data()
        grouped = self.data.groupby(groupby_args)[self.secondary_feature]

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

    def __call__(self):
        return self.binarize()


class Quantize(Group):
    """Quantize the data
    Args:
        feature (str.): Feature to be quantized
        data (pd.DataFrame): Data to be quantized
        n_bins (int, optional): Number of bins. Defaults to 10.
        
    Returns:
        np.ndarray: Quantized data

    Example:
        >>> data = pd.DataFrame(np.random.rand(10))
        >>> quantized_data = Quantize(data)
        >>> quantized_data()
    """
    def binarize(self):
        return pd.cut(self.data[self.feature].values, bins=self.bins)


class QuantizeDatetime(Group):
    bin_time_freq = ["D", "W", "M", "Q", "Y"]

    def binarize(self):
        groupby_args = pd.Grouper(key=self.feature, freq=self.bins)

        return self.data.groupby(groupby_args)[self.secondary_feature]

# @log_fun
@typechecked
def bin_and_agg(feature: str, data: pd.DataFrame, secondary_feature: str=None
                ,bins_boundaries: Union[np.array, str, bool]=None):
    # sourcery skip: remove-pass-elif
    """Aggregate feature according to bins. Use to Freedman-Diaconis Estimator 
    calculate bins [1].

    Args:
        feature (str): Feature binarized and aggregated, if a secondary_feature is not passed
        data (pd.DataFrame): Dataframe containing both features
        secondary_feature (str): Feature that is aggregated
        bins_boundaries (np.array or str or bool, optional): Array containing the bins. Defaults to True.

    Returns:
        pd.DataFrame: binarized and aggregated data

    References:
        [1] https://stats.stackexchange.com/questions/798/
            calculating-optimal-number-of-bins-in-a-histogram

    Example: #!TODO: tests
    """
    bin_edges_arg = ["auto", "fd", "doane", "scott", "stone", "rice"
                    , "sturges", "sqrt"]
    bin_time_freq = ["D", "W", "M", "Q", "Y"]

    secondary_feature = secondary_feature or feature

    if (bins_boundaries == True) and (data[feature].dtype == np.number):
        bins_boundaries = np.histogram_bin_edges(data[feature].values, 
                                                bins="auto")

    elif bins_boundaries in bin_edges_arg:
        bins_boundaries = np.histogram_bin_edges(data[feature].values, 
                                                bins=bins_boundaries)

    elif (not bins_boundaries) or (bins_boundaries in bin_time_freq) or \
        isinstance(bins_boundaries, np.ndarray):
        pass

    else:
        msg = f"Expected bins to be either {bin_edges_arg}, {bin_time_freq},\
                or bool. Got {bins_boundaries}."
        raise ValueError(msg)

    if isinstance(bins_boundaries, np.ndarray):
        groupby_args = pd.cut(data[feature], bins=bins_boundaries)

    elif bins_boundaries in bin_time_freq:
        groupby_args = pd.Grouper(key=feature, freq=bins_boundaries)

    else:
        groupby_args = feature

    grouped = data.groupby(groupby_args)[secondary_feature]

    return_dict = {"count": grouped.count
            ,"sum": grouped.sum
            ,"min": grouped.min
            ,"mean": grouped.mean
            ,"25%": grouped.quantile
            ,"50%": grouped.median
            ,"75%": grouped.quantile
            ,"max": grouped.max
            }

    output = return_dict["count"]().to_frame(name=f"count_{secondary_feature}")
    output[f"cum_count_{secondary_feature}"] = output[f"count_{secondary_feature}"].cumsum()
    output[f"proportions_{secondary_feature}"] = output[f"count_{secondary_feature}"]/output[f"count_{secondary_feature}"].sum()
    output[f"cum_proportions_{secondary_feature}"] = output[f"proportions_{secondary_feature}"].cumsum()

    if np.issubdtype(data[secondary_feature].dtype, np.number):
        output[f"min_{secondary_feature}"] = return_dict["min"]()
        output[f"mean_{secondary_feature}"] = return_dict["mean"]()
        output[f"25%_{secondary_feature}"] = return_dict["25%"](0.25)
        output[f"50%_{secondary_feature}"] = return_dict["50%"]()
        output[f"75%_{secondary_feature}"] = return_dict["25%"](0.75)
        output[f"max_{secondary_feature}"] = return_dict["max"]()

    return output


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
