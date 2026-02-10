from src.Features import transformed_data
import pandas as pd

data=transformed_data()

data=data.drop(columns=["MedHouseVal"])
