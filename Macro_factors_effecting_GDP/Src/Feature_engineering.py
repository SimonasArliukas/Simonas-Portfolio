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

def feature_engineering():
    """
    USE: Puts I(1) and I(0) variables in different tables and creates correcly rescaled GDP per capita variable.

    Returns
    --------
    Data: pd.DataFrame
        2 dataframes one with the relevant I(1) and the other with relevat I(0) variables.
    """
    data = pd.read_sql("SELECT * FROM GDP_inference_clean", engine)

    data.columns = ["date", "Labor Productivity", "Unemployment Rate", "Federal Funds Rate", "Inflation", "GDP",
                    "Population", "Investment", "Government Spending","Consumption","Net Exports"] ##Renaming the columns
    data["GDP_per_capita"] = data["GDP"]*1000000 / data["Population"] ##Rescaling GDP so that GDP and population are both in thousands

    exclude = ["date", "Unemployment Rate", "GDP", "Population"] ##Excluding I(0) and redundant variables
    data_i1 = data.loc[:, ~data.columns.isin(exclude)]

    re_order_cols = ['Government Spending','Labor Productivity',"Consumption", 'Investment', 'GDP_per_capita',"Federal Funds Rate", "Net Exports",'Inflation'] ##Reorder columns for choletsky decomposition
    data_i1_reorder = data_i1[re_order_cols]
    exogenous_variables = data[["Unemployment Rate"]]##Selecing exogenous variables for VECM
    return data_i1_reorder,exogenous_variables

feature_engineering()