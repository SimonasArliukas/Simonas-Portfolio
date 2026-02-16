from Src.Data_utils import differencing
from dotenv import load_dotenv
from sqlalchemy import create_engine
import os
import pandas as pd

load_dotenv()
##Connect to sql database
engine = create_engine(
    f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@"
    f"{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"
)




def feature_engineering():
    data = pd.read_sql("SELECT * FROM GDP_inference_clean", engine)
    
    data.columns = ["date", "Labor Productivity", "Unemployment Rate", "Federal Funds Rate", "Inflation", "GDP",
                    "Population", "Investment", "Government Spending"]
    data["GDP_per_capita"] = data["GDP"]*1000000 / data["Population"]

    exclude = ["date", "Unemployment Rate", "Federal Funds Rate", "GDP", "Population"]
    data_i1 = data.loc[:, ~data.columns.isin(exclude)]

    re_order_cols = ['Government Spending', 'Investment', 'Labor Productivity', 'GDP_per_capita', 'Inflation']
    data_i1_reorder = data_i1[re_order_cols]
    exogenous_variables = data[["Unemployment Rate", "Federal Funds Rate"]]
    return data_i1_reorder,exogenous_variables

feature_engineering()