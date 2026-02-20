from statsmodels.tsa.vector_ar.vecm import VECM
from Src.Feature_engineering import data
from Src.Feature_engineering import dummy_variables


dummy_variables_VECM=dummy_variables


model=VECM(data, k_ar_diff=1, coint_rank=2, deterministic='ci',exog=dummy_variables_VECM) ##Choosing order and lag from initial analysis
vecm_res=model.fit()
vecm_res.summary()


irf = vecm_res.irf(periods=10)
irf.plot(orth=True,response="GDP")
















