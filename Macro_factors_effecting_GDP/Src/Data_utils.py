import pandas as pd

def differencing(data):
    """
    USE: Computes differenced time series

    Parameters:
    -----------
    data : pandas.DataFrame
        Time series data you need to be differenced

    Returns:
    --------
    Differenced data : pandas.DataFrame
        Differenced time series data
    """
    data_empty = pd.DataFrame()

    for column in data.columns[1:]:
        diffed_series = data[column].diff()
        diffed_series.name = f"diffed_{column}"
        data_empty = pd.concat([data_empty, diffed_series], axis=1)

    return data_empty.reset_index(drop=True).dropna()




