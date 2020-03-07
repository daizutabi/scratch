import os

import pandas as pd

from ivory.utils.nnabla.utils import read_data


def read_monitor(path, melt=None, **kwargs):
    path = os.path.join(path, "*.series.txt")
    data = read_data(path, delimiter=" ", header=None)
    for key, df in data.items():
        key = os.path.basename(key).replace(".series.txt", "")
        df.columns = ["step", key]
        df.set_index("step", inplace=True)
    df = pd.concat(list(data.values()), axis=1)
    df.reset_index(inplace=True)

    if melt:
        columns = {
            column: column.split("_")[0]
            for column in df.columns
            if column.endswith("_" + melt)
        }

        df = df.rename(columns=columns)
        df = pd.melt(
            df,
            id_vars=["step"],
            value_vars=list(columns.values()),
            var_name="type",
            value_name=melt,
        )

    for key, value in kwargs.items():
        df[key] = value

    return df
