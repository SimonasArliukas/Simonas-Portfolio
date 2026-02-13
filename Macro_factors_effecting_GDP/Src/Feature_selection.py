from Src.Data_utils import log_differencing
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

data = pd.read_sql("SELECT * FROM GDP_inference_clean", engine)

log_differencing(data)
