from src.Features import selecting_data
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from src.Data_utils import Calculate_metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


data=selecting_data()

##Splitting into training and testing sets
X_train,X_test,Y_train,Y_test=train_test_split(data[0],data[1], test_size=0.2, random_state=42)

##Selecting features from feature selection step

X_train=X_train[['MedInc', 'Latitude', 'AveOccup', 'Longitude']]
X_test=X_test[['MedInc', 'Latitude', 'AveOccup', 'Longitude']]


#Linear model is just used as benchmark

linear_model=LinearRegression()
linear_model.fit(X_train,Y_train)
Y_pred=linear_model.predict(X_test)

Calculate_metrics(Y_test,Y_pred)

##Support vector regression

##First scaling and centering of features is an assumption for the model.
##It is necessary to scale and center after the split to not contaminate the testing set.

X_train_scaled=StandardScaler().fit_transform(X_train)
X_test_scaled=StandardScaler().fit_transform(X_test)

param_grid = {
    'kernel': ['linear',"rbf"], ##Controls the shape of the decision boundary rbf and poly can fit more complex decision boundaries
    'C': [0.1, 1,10], ##Regularization parameter on the loss function
    'epsilon': [0.01, 0.1] ,## epsilon insensitive tube defines the range where errors are ignored
    'gamma': ['scale']
}

##Compute grid search over all combinations of hyperparameters

grid_search = RandomizedSearchCV(
    SVR(),
    param_distributions=param_grid,
    n_iter=10,       # 10 random combinations
    cv=3,            # 3-fold CV for speed
    scoring='neg_mean_squared_error',
    n_jobs=-1,       # use all CPU cores
    random_state=42,
    verbose=1
)