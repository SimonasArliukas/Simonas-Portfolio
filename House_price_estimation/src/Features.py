from src.Data_utils import load_from_sql
from src.Data_utils import select_features
import numpy as np



data=load_from_sql("HOUSEPRICE")

##Transform the dependent variable

def transformed_data():
    data["LOG_MedHouseVal"]=np.log(data["MedHouseVal"]) #Median house value is always positive so log exists
    return data

#Only runs when executed directly
if __name__ == "__main__":
    ##Variable selection for the model. Using random forest because of non-linear relationships.
    def selecting_data():
        X = transformed_data().drop(["MedHouseVal", "LOG_MedHouseVal"], axis=1)  ##Predictive variables
        Y = transformed_data()["LOG_MedHouseVal"]  # Response variable
        return X, Y

    print(select_features(selecting_data()[0], selecting_data()[1], method="rf"))












