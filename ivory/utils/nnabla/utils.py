import glob
import sys


def set_args(line: str = ""):
    if '"' in line:
        raise NotImplementedError
    args = [arg for arg in line.split(" ") if arg]
    sys.argv = sys.argv[:1] + args


def read_data(path, **kwargs):
    import pandas as pd

    paths = glob.glob(path)

    def read(path):
        try:
            return pd.read_csv(path, **kwargs)
        except Exception:
            return None

    dfs = [(path, read(path)) for path in paths]
    return {path: df for path, df in dfs if df is not None}
