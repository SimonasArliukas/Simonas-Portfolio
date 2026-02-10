import os
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsRegressor
import shap
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


load_dotenv()


def load_from_sql(name_of_table):
    """
    USE: Imports a the sql table using env file into python.

    Parameters:
    -----------
    name_of_table: str
        Name of the table to import from sql database.

    Returns:
    -------
    SQL_Table: pandas.DataFrame
        Dataframe containing all data from sql table.
    """
    engine = create_engine(
        f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:"
        f"{os.getenv('MYSQL_PASSWORD')}@{os.getenv('MYSQL_HOST')}:"
        f"{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"
    )
    df=pd.read_sql_table(name_of_table,engine)
    return df


def select_features(X, y, method='lasso', top_k=5):

    """
    USE: Selects features from dataframe that best predict the response variable.

    Parameters
    ----------
    X: pandas.DataFrame
        Independent variables in the dataset that you want to select.
    y: pandas.Series
        Target variable in the dataset that you want to predict.
    method: str, default='lasso'
        Method of selecting features. Lasso, Random Forest or sequential feature selection.
    top_k:  int, default=5
        How many features to select in sequantial selection. Selects top k features.

    Returns
    --------
    selected_features: list
        Returns a list of top k selected features.

    """
    if method == 'lasso': ##Lasso has a property that does automatic variable selection
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X) ##Demeaning and scaling of the data to satisfy lasso assumption
        model = Lasso(alpha=0.1)
        model.fit(X_scaled, y)
        return X.columns[model.coef_ != 0].tolist()
    elif method == 'rf': ##Select variables most important in the prediction of baseline random forest using SHAP values
        #Regular importance show how many times variables were used for splits but SHAP shows how it influenced the prediction
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        sample_idx = X.sample(2000, random_state=42).index ##Sample 2000 random indeces for lower computation time
        X_sample = X.loc[sample_idx]
        y_sample = y.loc[sample_idx]

        model.fit(X_sample, y_sample)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Global importance: mean absolute SHAP value
        shap_importance = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=X_sample.columns
        )

        return (
            shap_importance[shap_importance > 0.1]
            .sort_values(ascending=False)
            .index
            .tolist()
        )
    elif method == 'sequential': ##Sequential selection
        knn = KNeighborsRegressor(n_neighbors=5)
        sfs = SequentialFeatureSelector(knn,n_features_to_select=top_k,direction='forward')
        sfs = sfs.fit(X, y)
        return list(sfs.support_)

def mean_absolute_percentage_error(Y_TRUE, Y_PRED):
    """
    USE: Calculates the mean absolute percentage error.

    Parameters
    ----------
    Y_TRUE: Pandas.Series
        True response values taken from the dataset.
    Y_PRED: Pandas.Series
        Predicted response values taken from the model.
    Returns
    -----------
    mean_absolute_percentage_error: float
        Mean absolute percentage error.

    """
    y_true, y_pred = np.array(Y_TRUE), np.array(Y_PRED)
    # Avoid division by zero
    non_zero_idx = y_true != 0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

def Calculate_metrics(Y_test,Y_pred):
    """
    USE: Calculates root mean square error, mean absolute error and mean absolute percentage error.

    Parameters
    ----------
    Y_test: pandas.Series
        True response values taken from the dataset.
    Y_pred: pandas.Series
        Predicted response values taken from the model.

    Returns
    -------
    metrics: tuple
        Retruns mean square error, mean absolute error, and mean absolute percentage error.
    """
    rmse = mean_squared_error(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    mape = mean_absolute_percentage_error(Y_test, Y_pred)

    return dict(RMSE=rmse, MAE=mae, MAPE=float(mape))











