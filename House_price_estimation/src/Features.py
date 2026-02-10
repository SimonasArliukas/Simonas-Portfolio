from src.Data_utils import load_from_sql
from src.Data_utils import select_features
import numpy as np

data=load_from_sql("HOUSEPRICE")

##Transform the dependent variable

data["LOG_MedHouseVal"]=np.log(data["MedHouseVal"]) #Median house value is always positive so log exists

##Variable selection for the model. Using random forest because of non-linear relationships.

X=data.drop(["MedHouseVal","LOG_MedHouseVal"],axis=1) ##Predictive variables
Y=data["LOG_MedHouseVal"] #Response variable

print(select_features(X,Y,method="rf"))



