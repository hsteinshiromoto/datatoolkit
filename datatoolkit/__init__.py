"""Manipulation of data

Modules exported by this package:

- `eda`: Provides several functions for data manipulation.
- `hypothesis`: Provides several functions for different hypothesis tests.
- `mock_dataset`: Provides generator of data frame containing random data.
- `model_selection`: Provides cross validation using Bayesian optimization, and cost functionals.
- `visualize`: Provides data visualization tools.
"""

__version__ = "0.2.5"

from datatoolkit.eda import Group, Numerical, Summarize
from datatoolkit.hypothesis import SingleSampleTest, TwoSampleTest
from datatoolkit.mock_dataset import DataTypes, MockData
from datatoolkit.model_selection import (
    BayesianSearchCV,
    ClassificationCostFunction,
    CostFunction,
)
from datatoolkit.utils import MostFrequent, Quantize, QuantizeDatetime
from datatoolkit.visualize import (
    dash_line,
    graphplot,
    heatmap_4d,
    hist_box,
    line_bar_plot,
)
