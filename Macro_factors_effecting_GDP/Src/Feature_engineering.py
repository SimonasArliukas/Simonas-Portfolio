from dotenv import load_dotenv
from sqlalchemy import create_engine
import os
import pandas as pd
import numpy as np


load_dotenv()
##Connect to sql database
engine = create_engine(
    f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@"
    f"{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"
)

data = pd.read_sql("SELECT * FROM GDP_inference_clean", engine)

def renaming(data):
    data.columns = ["date", "Labor Productivity", "Unemployment Rate", "Federal Funds Rate", "Inflation", "GDP",
                    "Population", "Investment", "Government Spending","Consumption","Net Exports"] ##Renaming the columns
    return data

data=renaming(data)


def connverting_to_log(data) :
    cols_to_log = ['GDP', 'Investment', 'Consumption', 'Government Spending']

    for i in cols_to_log:
        data[i]=np.log(data[i])

    return data

data=connverting_to_log(data)

def exclusion_reordering(data):
    """
    USE: Excludes unnecessary columns and reorders data for choletsky decomposition

    Paramaters:
    -----------
    data: pd.DataFrame
        Data that you would like to exclude and reorder

    Returns:
    ---------
    data_01_order: pd.DataFrame
        Data with excluded and reordered columns

    """
    exclude = ["date", "GDP_per_capita", "Population","Net Exports"]
    data_i1 = data.loc[:, ~data.columns.isin(exclude)]

    re_order_cols = [
        "Labor Productivity",
        "Government Spending",
        "GDP",
        "Consumption",
        "Investment",
        "Unemployment Rate",
        "Inflation",
        "Federal Funds Rate"
    ]
    ##Reorder columns for choletsky decomposition
    data_i1_reorder = data_i1[re_order_cols]
    data_i1_reorder.index= data["date"]

    return data_i1_reorder

data=exclusion_reordering(data)




dates = pd.date_range(start="1985-01-01", end="2019-12-31", freq='QS') ##The entire period
dummy_variables = pd.DataFrame(index=dates)


def add_dummies(df, break_dates=["1986-04-01","1986-07-01","1987-04-01","1987-07-01","2008-07", "2008-10"]):
    """
    USE: Adds dummy variables for specific break dates

    Paramaters:
    -----------
    df: pd.DataFrame
        Dataframe where you would like to add dummy variables
    break_dates: list
        List of break dates

    Returns:
    ---------
    df: pd.DataFrame
        Dataframes with added dummy variables
    """
    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)

    for bd in break_dates:
        col_name = f"dummy_{bd.replace('-', '_')}" ##Specify column names
        # Compare year and month only
        target = pd.to_datetime(bd) ##convert date to datetime
        df[col_name] = ((df.index.year == target.year) & (df.index.month == target.month)).astype(int) ##Add a dummy variable to specific date and month

    return df

dummy_variables = add_dummies(dummy_variables)

