"""
Kaggle Titanic Challenge
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# load the datasets
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# transforming the data
X_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
y_cols = ["Survived"]
X_train = train.loc[:, X_cols]
X_test = test.loc[:, X_cols]
y_train = np.ravel(train.loc[:, y_cols])


def nans_via_knn(data):
    """
    Fills in nans in all columns using ineference from the most similar rows
    """
    rows_w_nan = np.arange(data.shape[0])[data.isnull().any(axis=1).values]
    rows_wo_nan = np.arange(data.shape[0])[~data.isnull().any(axis=1).values]
    # subset of observavtions without any nans
    base = data.iloc[rows_wo_nan, :]
    # standardization
    means = np.mean(base, axis=0)
    stds = np.std(base, axis=0)
    base = (base - means)/stds
    # fill in each row with nan
    for i in rows_w_nan:
        cols_w_nan = np.arange(data.shape[1])[data.iloc[i, :].isnull()]
        cols_wo_nan = np.arange(data.shape[1])[~data.iloc[i, :].isnull()]
        # find closest observations
        i_standardized = (data.iloc[i, cols_wo_nan].values - means[cols_wo_nan])/stds[cols_wo_nan]
        dists = base.iloc[:, cols_wo_nan].values - i_standardized[np.newaxis, :]
        dists = np.sum(np.square(dists), axis=1)
        closest = np.argsort(dists)[:5]
        # fill in each column with nan using median of the closest observations
        for c in cols_w_nan:
            data.iloc[i, c] = np.median(base.iloc[closest, c])*stds[c] + means[c]
    return data


def prepare_features(data):
    # Sex (=1 if male)
    data["Male"] = (data["Sex"] == "male") * 1
    # dummy variables for point of departure (if all 0 then it is nan)
    data["Embarked_S"] = (data["Embarked"]=="S")*1
    data["Embarked_C"] = (data["Embarked"]=="C")*1
    data["Embarked_Q"] = (data["Embarked"]=="Q")*1
    # drop the encoded cols
    data = data.drop(["Sex", "Embarked"], axis=1)
    # replace age nans with -1 and let the model know
    data["Age_nan"] = (np.isnan(data["Age"]))*1
    data["Age"][np.isnan(data["Age"])]=-1
    # fill-in nans if there are any
    data = nans_via_knn(data)
    # scaling
    data = pd.DataFrame(
            data=StandardScaler().fit_transform(data),
            columns=list(data)
            )
    return data

# prepare the data for modelling
X_train = prepare_features(X_train)
X_test = prepare_features(X_test)
# =============================================================================
# GRID SEARCH OVER DIFFERENT MODEL T
# =============================================================================
from skopt.space import Real, Categorical, Integer
from sklearn.svm import LinearSVC, SVC

linsvc_search = {
    'model': [LinearSVC(max_iter=1000)],
    'model__C': (1e-6, 1e+6, 'log-uniform'),
}

svc_search = {
    'model': Categorical([SVC()]),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'model__degree': Integer(1, 8),
    'model__kernel': Categorical(['linear', 'poly', 'rbf']),
}

opt = BayesSearchCV(
    pipe,
    [(svc_search, 1), (linsvc_search, 1)], # (parameter space, # of evaluations)
    cv=5
)

opt.fit(X_train, y_train, callback=status_print)


# =============================================================================
# PREDICTIONS ON THE COMPETITION TEST SET
# =============================================================================
y_pred = df_results.iloc[2, 1].predict(X_test)
sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": y_pred})
# save the predictions
sub.to_csv("data/submission.csv",  index=False)
# =============================================================================
# END OF FILE
# =============================================================================
