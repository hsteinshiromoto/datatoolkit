import functools
import operator
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Union

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as ss
from deprecated import deprecated
from typeguard import typechecked

PROJECT_ROOT = Path(
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)

sys.path.append(PROJECT_ROOT)

# from tests.mock_dataset import mock_dataset
# from src.make_logger import log_fun


def flatten(array: Iterable[Iterable]) -> Iterable:
    """Flattens nested iterable

    Args:
        array (Iterable[Iterable]): Nested iterable

    Yields:
        Iterable: Flattened iterable

    Example:
        >>> a = [1, ['a', 'b']]
        >>> list(flatten(a))
        [1, 'a', 'b']
    """
    for x in array:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


# @log_fun
@typechecked
def make_pivot(
    feature: str, index: str, column: str, data: pd.DataFrame, groupby_args: list = None
):
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

    grouped = (
        data.groupby(groupby_args)[feature].count().to_frame(name=f"count_{feature}")
    )

    try:
        grouped[f"mean_{feature}"] = data.groupby(groupby_args)[feature].mean()

    except ValueError:
        if np.issubdtype(data[feature].dtype, np.number):
            msg = f"Expected feature {feature} to of data type numerical. Got {data[feature].dtype}."
            raise (msg)

        raise

    grouped.reset_index(inplace=True)
    grouped.sort_values(by=[index, column], inplace=True, ascending=False)

    pivot_count = pd.pivot(
        grouped, index=index, columns=column, values=f"count_{feature}"
    )
    pivot_mean = pd.pivot(
        grouped, index=index, columns=column, values=f"mean_{feature}"
    )

    pivot_count.sort_index(inplace=True, ascending=False)
    pivot_mean.sort_index(inplace=True, ascending=False)

    return pivot_count, pivot_mean


# @log_fun
@typechecked
class MostFrequent:
    """Truncates data according to the proportion of a categorical column

    Args:
        array (Iterable): Array of categorical values
        top_pct_obs (float, optional): Percentage of observations to use. Defaults to 0.8.
        top_pct_cat (float, optional): Percentage of categories to use. Defaults to 0.2.

    Returns:
            (Iterable, pd.DataFrame): 1d array with most frequent categories and
                                        summary statistics

    References:
        [1] https://hsteinshiromoto.github.io/posts/2020/06/25/find_row_closest_value_to_input

    Example:
        >>> x = [[i]*j for j,i in zip([50, 25, 12, 6, 3, 2, 2], range(1, 7))]
        >>> x = functools.reduce(operator.iconcat, x, []) # Flat the list
        >>> mf = MostFrequent(x)
        >>> output, stats = mf()
        >>> print(stats[["category", "n_observations_proportions", "cum_n_observations_proportions"]])
                   category  n_observations_proportions  cum_n_observations_proportions
        0                 1                    0.510204                        0.510204
        1                 2                    0.255102                        0.765306
        2  other categories                    0.489796                        1.000000
    """

    @typechecked
    def __init__(
        self, data: Iterable, top_pct_obs: float = 0.8, top_pct_cat: float = 0.2
    ):
        self.data = data
        self.top_pct_obs = top_pct_obs
        self.top_pct_cat = top_pct_cat

    def fit(self):
        """
        Make statistical summary of the data

        Returns:
            (None)
        """
        unique, counts = np.unique(self.data, return_counts=True)
        self.grouped = pd.DataFrame.from_dict(
            {"category": unique, "n_observations": counts}
        )
        self.grouped.sort_values(by="n_observations", ascending=False, inplace=True)
        self.grouped["n_observations_proportions"] = (
            self.grouped["n_observations"] / self.grouped["n_observations"].sum()
        )
        self.grouped["cum_n_observations_proportions"] = self.grouped[
            "n_observations_proportions"
        ].cumsum()
        self.grouped["cum_n_categories_proportions"] = np.linspace(
            1.0 / float(self.grouped.shape[0]), 1, self.grouped.shape[0]
        )
        self.grouped.reset_index(drop=True, inplace=True)

    def transform(self):
        """
        Locate the category that is closest to the top_pct_cat or observation that
        is close to top_pct_obs proportion

        Raises:
            ValueError: Raise if floats are not positive

        Returns:
            (pd.DataFrame): Categories and summary data set
        """

        if (self.top_pct_obs > 0) & (self.top_pct_cat > 0):
            subset = (
                self.grouped["cum_n_observations_proportions"]
                + self.grouped["cum_n_categories_proportions"]
            )
            threshold = self.top_pct_obs + self.top_pct_cat

            # Get row containing values closed to a value [1]
            idx = subset.sub(threshold).abs().idxmin()

        elif self.top_pct_obs > 0:
            idx = (
                self.grouped["cum_n_observations_proportions"]
                .sub(self.top_pct_obs)
                .abs()
                .idxmin()
            )

        elif self.top_pct_cat > 0:
            idx = (
                self.grouped["cum_n_categories_proportions"]
                .sub(self.top_pct_cat)
                .abs()
                .idxmin()
            )

        else:
            msg = (
                f"Expected top_pct_obs or top_pct_cat to be positive. "
                f"Got top_pct_obs={self.top_pct_obs} and top_pct_cat={self.top_pct_cat}"
            )
            raise ValueError(msg)

        self.grouped.loc[idx + 1, "category"] = "other categories"
        self.grouped.loc[idx + 1, "cum_n_observations_proportions"] = 1

        self.grouped.loc[idx + 1, "n_observations"] = self.grouped.loc[
            idx:, "n_observations"
        ].sum()
        self.grouped.loc[idx + 1, "n_observations_proportions"] = self.grouped.loc[
            idx:, "n_observations_proportions"
        ].sum()

        return (
            self.grouped.loc[: idx + 1, "category"].values,
            self.grouped.loc[: idx + 1, :],
        )

    def __call__(self):
        self.fit()
        return self.transform()


def make_graph(
    nodes: Iterable, M: np.ndarray, G: nx.classes.digraph.DiGraph = nx.DiGraph()
):
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
                G.add_edge(
                    origin_node,
                    destination_node,
                    weight=M[i, j],
                    label=f"{M[i, j]:0.02f}",
                )

    return G


def make_distributions(
    parameters: dict[str, dict[str, Union[str, tuple, Iterable]]]
) -> dict[str, Union[Callable, Iterable]]:
    """Returns SciPy statistical distribution objects.

    Args:
        parameters (dict[str, dict[str, Union[str, tuple, Iterable]]]): Distribution parameters.

    Returns:
        dict[str, Union[Callable, Iterable]]: Distribution objects

    Example:
        >>> parameters = {
        ...    "min_weight_fraction_leaf": {"distribution": "norm", "args": (0.25, 0.01)},
        ...    "criterion": {"distribution": "choice", "args": ["gini", "entropy"]} }
        >>> distr_dict = make_distributions(parameters)
        >>> distr_dict["min_weight_fraction_leaf"].stats(moments='mvsk')
        (array(0.25), array(0.0001), array(0.), array(0.))
        >>> distr_dict["criterion"]
        ['gini', 'entropy']
    """
    params_dict = {}
    for parameter, description in parameters.items():
        try:
            distr = getattr(ss, description["distribution"])

        except AttributeError:
            distribution = description["args"]

        else:
            distribution = distr(*description["args"])

        params_dict[parameter] = distribution

    return params_dict
