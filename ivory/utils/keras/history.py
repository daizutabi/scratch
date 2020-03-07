import altair as alt
import pandas as pd


def history_to_dataframe(history, key):
    if isinstance(history, dict):
        dfs = []
        for name, history_ in history.items():
            df = history_to_dataframe(history_, key)
            df["name"] = name
            dfs.append(df)
        return pd.concat(dfs)

    if isinstance(key, list):
        df = None
        for key_ in key:
            df_ = history_to_dataframe(history, key_)
            df = df_ if df is None else df.merge(df_)
        return df

    dfs = []
    for key_ in [key, "val_" + key]:
        df = pd.DataFrame({"epoch": history.epoch, key: history.history[key_]})
        if key_.startswith("val_"):
            df["type"] = "validation"
        else:
            df["type"] = "train"
        dfs.append(df)
    return pd.concat(dfs)


def plot_history(history, key, **kwargs):
    df = history_to_dataframe(history, key)
    chart = alt.Chart(df).encode(x="epoch", y=key, **kwargs)
    return chart.mark_line().encode(detail="type") + chart.mark_point().encode(
        shape="type"
    ).properties(width=260, height=200)
