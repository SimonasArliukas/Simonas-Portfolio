from sqlalchemy import create_engine
from fredapi import Fred
import os
from dotenv import load_dotenv
import pandas as pd


load_dotenv()

fred = Fred(api_key=os.getenv('FRED_API_KEY'))

engine = create_engine(
    f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@"
    f"{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"
)


def Scrape_from_Fred(variables: list):
    """
    USE: Functions scrapes data from FRED API and puts it into a dataframe

    Parameters
    ----------
    variables : list
        List of variables to scrape (variable names must correspond to Fred time series names)

    Return
    ------------
    Scraped data: pd.DataFrame
        Returns time series dataframe scraped from FRED API
    """

    data = pd.DataFrame()

    for variable in variables:
        temp = fred.get_series(variable)
        temp_df = temp.reset_index()  ##Reset index so the first column is date
        temp_df.columns = ["date", variable]  ##Sets the first column is date so we can merge on date

        if data.empty:
            data = temp_df
        else:
            data = pd.merge(data, temp_df, on="date", how="outer")

    return data


Time_series_data = Scrape_from_Fred(
    ["PRS85006091", "UNRATE", "DFF", "CORESTICKM159SFRBATL", "gdp", "POPTHM","GPDIC1","FGEXPND","PCE","NETEXP"])

Time_series_data.to_sql(
    name="GDP_inference",
    con=engine,
    if_exists="replace",
    index=False
)

Time_series_data.to_csv("GDP_inference.csv")

