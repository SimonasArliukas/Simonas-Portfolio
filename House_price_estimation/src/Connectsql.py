import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

House_price=pd.read_csv('fetch_california_housing.csv') ##Takes the house price dataset

engine = create_engine(
    f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"
)

House_price.to_sql(
    name="HOUSEPRICE",
    con=engine,
    if_exists="replace",
    index=False
) ##Puts the house price dataset into an sql table.



