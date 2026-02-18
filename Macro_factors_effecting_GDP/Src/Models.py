from statsmodels.tsa.vector_ar.vecm import VECM
from Src.Feature_engineering import feature_engineering
import numpy as np


data_i1_reorder,exogenous_variables= feature_engineering() ##I(1) variables are in the model while I(0) are exogenous


model=VECM(data_i1_reorder, k_ar_diff=3, coint_rank=2, deterministic='ci',exog=exogenous_variables) ##Choosing order and lag from initial analysis
vecm_res=model.fit()
vecm_res.summary()

irf = vecm_res.irf(periods=10)
irf.plot(orth=True,response="GDP_per_capita")


















