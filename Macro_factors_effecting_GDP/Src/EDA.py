import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from Src.Data_utils import differencing
from statsmodels.tsa.stattools import adfuller
from Src.Data_utils import cointegration
from sqlalchemy import create_engine

load_dotenv()
##Connect to sql database
engine = create_engine(
    f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@"
    f"{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"
)

##Load the data from SQL

data = pd.read_sql("SELECT * FROM GDP_inference_clean", engine)

data.columns = ["date", "Labor Productivity", "Unemployment Rate", "Federal Funds Rate", "CPI", "GDP", "Population"]
data["GDP_per_capita"] = data["GDP"] / data["Population"]

fig, axes = plt.subplots(4, 2, figsize=(15, 6), sharex=True)
axes = axes.flatten()
for i, column in enumerate(data.columns[1:]):
    axes[i].plot(data["date"], data[column], label=column, color='tab:blue')
    axes[i].set_ylabel(column)
    axes[i].legend(loc='upper left')

plt.xlabel("Date")
plt.tight_layout()
plt.show()
data

##Calculates unit roots to formally test stationarity
def dickey_fuller_test(data_test):
    """
    USE: Computes Dickey Fuller Test for each time series

    Parameters:
    -----------
    data_test: pandas dataframe
        Dataframe containing time series data

    Returns:
    --------
    Dickey Fuller : Dict
        Dickey Fuller test statistic and p-value for each time series

    """
    if "date" in data_test.columns:
        data_test = data_test.drop(["date"], axis=1)
    for i in data_test.columns:
        adf_result = adfuller(data_test[i], regression="ct")
        print(f"{i}, ADF Statistic: {adf_result[0]}, p-value: {adf_result[1]}")


dickey_fuller_test(data)

cointegration(data)

data.to_csv("GDP_inference_clean.csv")
