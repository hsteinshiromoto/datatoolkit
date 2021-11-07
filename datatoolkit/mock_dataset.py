from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from typeguard import typechecked
from enum import Enum, auto


class DataTypes(Enum):
    """Supermarket

    Args:
        Enum ([type]): [description]

    Returns:
        [type]: [description]
    """

    float = auto()
    int = auto()
    cat = auto()
    bool = auto()
    str = auto()
    dt = auto()


class MockData:
    """
    Create mock pandas dataframe

    # TODO: Test metadata
    Example:
        >>> md = MockData()
        >>> df = md()
        >>> meta_data = md.make_meta_data()
    """

    @typechecked
    def __init__(self, specs_dict: dict=None, n_rows: int=100
                , n_cols: int=2
                , n_nas: float=0.05):
        """
        Args:
            specs_dict (dict, optional): Specification for data frame construction. Defaults to None.
            n_rows (int, optional): Number of rows in the data frame. Defaults to 100.
            n_cols (int, optional): Number of columns of each type. Defaults to 2.
            n_nas (float, optional): Proportion of missing values in each column. Defaults to 0.05.
        """
        self.specs_dict = specs_dict or self._make_specs(n_rows, n_cols, n_nas)


    @staticmethod
    def _make_specs(n_rows: int, n_cols: int, n_nas: float) -> dict:
        """
        Generate specification for data frame construction

        Args:
            See __init__

        Returns:
            dict: Specification for data frame construction
        """
        return {col: [n_rows, n_cols, n_nas] for col in 
                    [DataTypes.float, DataTypes.int ,DataTypes.cat
                    ,DataTypes.bool, DataTypes.str, DataTypes.dt]
                }


    def build_column(self, dtype: DataTypes, col_spec: list):
        if dtype is DataTypes.float:
            return np.random.rand(col_spec[0], 1).flatten()

        elif dtype is DataTypes.int:
            return np.random.randint(np.random.randint(1e6), size=col_spec[0]).flatten()

        elif dtype is DataTypes.cat:
            return ["".join(category.flatten()) for category in np.random.choice(["A", "B", "C", "D"], size=(col_spec[0], 3))]

        elif dtype is DataTypes.bool:
            return [bool(item) for item in np.random.randint(2, size=col_spec[0])]

        elif dtype is DataTypes.str:
            return ["".join(category.flatten()) for category in np.random.choice(
                        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "X", "Y", "W", "Z"], size=(col_spec[0], col_spec[0]))]
        
        elif dtype is DataTypes.dt:
            return list(
                    pd.date_range(datetime.today(), periods=col_spec[0]))


    def make_dataframe(self):

        values = {}
        for col_type, col_spec in self.specs_dict.items():
            for count in range(col_spec[1]):
                    values[f"{col_type.name}_{count}"] = self.build_column(col_type, col_spec)

        self.data = pd.DataFrame.from_dict(values)

        return self.add_nans()


    def add_nans(self):
        for col_type, col_spec in self.specs_dict.items():
            for col in [col for col in self.data.columns.values if col_type.name in col]:
                mask = self.data[col].sample(frac=col_spec[2]).index
                self.data.loc[mask, col] = np.nan

        return self.data


    def make_meta_data(self):

        meta_data_dtype_map = {"float": "float", "int": "int", "cat": "category",
                        "str": "str", "dt": "datetime64[ns]", "bool": "bool"}

        meta_data_dict = {"column_name": [], "python_dtype": []}
        for col in self.data.columns.values:
            meta_data_dict["column_name"].append(col)
            meta_data_dict["python_dtype"].append(
                meta_data_dtype_map[col.split("_")[0]])

        self.meta_data = pd.DataFrame.from_dict(meta_data_dict)

        return self.meta_data


    def __call__(self, return_meta_data: bool=False):
        if return_meta_data:
            return self.make_dataframe(), self.make_meta_data()

        return self.make_dataframe()