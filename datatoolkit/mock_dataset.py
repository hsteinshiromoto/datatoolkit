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


class MockData(DataTypes):
    def __init__(self, specs_dict: dict=None):
        self.specs_dict = specs_dict


    def make_specs(self, n_rows: int=100, n_cols: int=np.random.randint(1, 4)
                ,n_nas: float=np.random.rand()):
        self.specs_dict = {col: [n_rows, n_cols, n_nas] for col in 
                    [DataTypes.float, DataTypes.int, DataTypes.cat, DataTypes.bool, DataTypes.str, DataTypes.dt]}


    def make_dataframe(self):

        values = {}
        for col_type, col_spec in self.specs_dict.items():
            if col_type is DataTypes.float:
                for count in range(col_spec[1]):
                    values[f"{col_type.name}_{count}"] = np.random.rand(
                        col_spec[0], 1).flatten()

            elif col_type is DataTypes.int:
                for count in range(col_spec[1]):
                    values[f"{col_type.name}_{count}"] = np.random.randint(
                        np.random.randint(1e6), size=col_spec[0]).flatten()

            elif col_type is DataTypes.cat:
                for count in range(col_spec[1]):
                    values[f"{col_type.name}_{count}"] = ["".join(category.flatten(
                    )) for category in np.random.choice(["A", "B", "C", "D"], size=(col_spec[0], 3))]

            elif col_type is DataTypes.bool:
                for count in range(col_spec[1]):
                    values[f"{col_type.name}_{count}"] = [
                        bool(item) for item in np.random.randint(2, size=col_spec[0])]

            elif col_type is DataTypes.str:
                for count in range(col_spec[1]):
                    values[f"{col_type.name}_{count}"] = ["".join(category.flatten()) for category in np.random.choice(
                        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "X", "Y", "W", "Z"], size=(col_spec[0], col_spec[0]))]

            elif col_type is DataTypes.dt:
                values[f"{col_type.name}_{count}"] = list(
                    pd.date_range(datetime.today(), periods=col_spec[0]))

        self.data = pd.DataFrame.from_dict(values)

        self.add_nas()

        return self.data


    def add_nas(self):
        for col_type, col_spec in self.specs.items():
            for col in [col for col in self.data.columns.values if col_type in col]:
                mask = self.data[col].sample(frac=col_spec[2]).index
                self.data.loc[mask, col] = np.nan


    def make_meta_data(self):

        meta_data_dtype_map = {"float": "float", "int": "int", "cat": "category",
                        "str": "str", "dt": "datetime64[ns]", "bool": "bool"}

        meta_data_dict = {"column_name": [], "python_dtype": []}
        for col in self.data.columns.values:
            meta_data_dict["column_name"].append(col)
            meta_data_dict["python_dtype"].append(
                meta_data_dtype_map[col.split("_")[0]])

        self.meta_data = pd.DataFrame.from_dict(meta_data_dict)


    def get_meta_data(self):
        self.make_meta_data()

        return self.meta_data


    def __call__(self, return_meta_data: bool=False):
        if self.specs_dict is None:
            self.make_specs()

        if return_meta_data:
            return self.make_dataframe(), self.get_meta_data()
        return self.make_dataframe()



@typechecked
def mock_dataset(specs: dict = None, meta_data: bool = False):
    """
    Create mock pandas dataframe

    Args:
        specs (dict, optional): Specifications of the data frame. Defaults to None.
        meta_data (bool, optional): Return meta data as pandas dataframe

    Returns:
        pd.DataFrame: mock pandas dataframe

    # TODO: Test metadata
    Example:
        >>> specs = {"float": [100, 1, 0.05] \
                    ,"int": [100, 1, 0.025] \
                    ,"categorical": [100, 1, 0.1] \
                    ,"bool": [100, 1, 0] \
                    ,"str": [100, 1, 0] \
                    ,"datetime": [100, 1, 0] \
                    }
        >>> df, meta_data = mock_dataset(specs, True)
    """
    # 1. Build specs, in case needed
    if not specs:
        # Format of specs dict: {data_type: [nrows, ncols, nnulls]}
        specs = {"float": [100, np.random.randint(1, 4), np.random.rand()], "int": [100, np.random.randint(1, 4), np.random.rand()], "categorical": [100, np.random.randint(1, 4), 0.75], "bool": [100, np.random.randint(1, 4), np.random.rand()], "str": [100, np.random.randint(1, 4), np.random.rand()], "datetime": [100, np.random.randint(1, 4), np.random.rand()]
                 }

    # 2. Build values of the data frame
    values = {}
    for col_type, col_spec in specs.items():
        if col_type == "float":
            for count in range(col_spec[1]):
                values[f"{col_type}_{count}"] = np.random.rand(
                    col_spec[0], 1).flatten()

        elif col_type == "int":
            for count in range(col_spec[1]):
                values[f"{col_type}_{count}"] = np.random.randint(
                    np.random.randint(1e6), size=col_spec[0]).flatten()

        elif col_type == "categorical":
            for count in range(col_spec[1]):
                values[f"{col_type}_{count}"] = ["".join(category.flatten(
                )) for category in np.random.choice(["A", "B", "C", "D"], size=(col_spec[0], 3))]

        elif col_type == "bool":
            for count in range(col_spec[1]):
                values[f"{col_type}_{count}"] = [
                    bool(item) for item in np.random.randint(2, size=col_spec[0])]

        elif col_type == "str":
            for count in range(col_spec[1]):
                values[f"{col_type}_{count}"] = ["".join(category.flatten()) for category in np.random.choice(
                    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "X", "Y", "W", "Z"], size=(col_spec[0], col_spec[0]))]

        elif col_type == "datetime":
            values[f"{col_type}_{count}"] = list(
                pd.date_range(datetime.today(), periods=col_spec[0]))

    df = pd.DataFrame.from_dict(values)

    # 3. Add nulls according to the proportion specified
    for col_type, col_spec in specs.items():
        for col in [col for col in df.columns.values if col_type in col]:
            mask = df[col].sample(frac=col_spec[2]).index
            df.loc[mask, col] = np.nan

    if not meta_data:
        return df

    # 4. Get meta data

    meta_data_dtype_map = {"float": "float", "int": "int", "categorical": "category",
                           "str": "str", "datetime": "datetime64[ns]", "bool": "bool"}

    meta_data_dict = {"column_name": [], "python_dtype": []}
    for col in df.columns.values:
        meta_data_dict["column_name"].append(col)
        meta_data_dict["python_dtype"].append(
            meta_data_dtype_map[col.split("_")[0]])

    meta_data = pd.DataFrame.from_dict(meta_data_dict)

    return df, meta_data


if __name__ == "__main__":
    data, meta_data = mock_dataset(meta_data=True)
    print("End")
