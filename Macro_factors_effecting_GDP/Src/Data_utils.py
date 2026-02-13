import numpy as np
from statsmodels.tsa.stattools import coint
from itertools import combinations
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


def cointegration(data_coint, significance=0.05):
    """
    USE: Computes cointegration test between every combination of time series

    Parameters:
    ----------
    data_coint : pandas.DataFrame
        Time series data you need to be cointegrated

    Returns:
    --------
    p-values and cointegrated combinations: pd.DataFrame and tuple
        P-values of the cointegration test and cointegrated combinations
    """
    coint_matrix = pd.DataFrame(index=data_coint.columns, columns=data_coint.columns, dtype=float)

    pairs = []
    if "date" in data_coint.columns:
        data_coint = data_coint.drop("date", axis=1)

    for col1, col2 in combinations(data_coint.columns, 2):
        _, pvalue, _ = coint(data_coint[col1], data_coint[col2])
        coint_matrix.loc[col1, col2] = pvalue
        coint_matrix.loc[col2, col1] = pvalue

        if pvalue < significance:
            pairs.append((col1, col2))

    for col in data_coint.columns:
        coint_matrix.loc[col, col] = None
    return coint_matrix, pairs
