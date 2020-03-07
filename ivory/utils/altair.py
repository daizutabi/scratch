import altair as alt


def bar_from_series(series, column_name, index_name):
    series.name = column_name
    df = series.to_frame()
    df.index.name = index_name
    df.reset_index(inplace=True)
    y = alt.Y(index_name, sort=alt.EncodingSortField(field=column_name, op="values"))
    return alt.Chart(df).mark_bar().encode(x=column_name, y=y)
