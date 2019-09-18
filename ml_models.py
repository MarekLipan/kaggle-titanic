"""
Machine Learning Models
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from copy import deepcopy
import multiprocessing

# turn off the deprecation warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




# BAYESIAN GRID SEARCH TUNNER
class Tunner:
    """
    Tunner object used for finding optimal set of hyperparameters for a given
    model respective to the given data using Bayesian Grid Search.
    """

    def __init__(self, estimator, search_spaces, n_iter):
        self.opt = BayesSearchCV(
            estimator=estimator,
            search_spaces=search_spaces,
            scoring="accuracy",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=40),
            #n_jobs=multiprocessing.cpu_count(),
            n_jobs=1,
            n_iter=n_iter,
            verbose=0,
            refit=True,
            random_state=43
            )


# =============================================================================
# CALLBACKS
# =============================================================================
def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""

    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(t.opt.cv_results_)

    # Get current parameters and the best parameters
    # best_params = pd.Series(opt.best_params_)
    print(
        "Iteration #{} Model: {} Accuracy: {}".format(
            len(all_models),
            model_name,
            np.round(t.opt.best_score_, 4)
        )
    )


# main function
no_iter  = 50

# create dict to store info about different types of models
models = {"model_name" : [], "raw_estimator" : [], "search_space" : []}
# =============================================================================
# DIFFERENT TYPES OF MODELS
# =============================================================================
# XGBoost
models["model_name"].append("XGBClassifier")
models["raw_estimator"].append(XGBClassifier(n_jobs=1, objective="multi:softmax", silent=1, tree_method="approx", num_class=2))
models["search_space"].append({
        "learning_rate": (0.01, 1.0, "log-uniform"),
        "max_depth": (1, 50),
        "max_delta_step": (0, 20),
        "subsample": (0.01, 1.0, "uniform"),
        "colsample_bytree": (0.01, 1.0, "uniform"),
        "colsample_bylevel": (0.01, 1.0, "uniform"),
        "reg_lambda": (1e-9, 1000, "log-uniform"),
        "reg_alpha": (1e-9, 1.0, "log-uniform"),
        "gamma": (1e-9, 0.5, "log-uniform"),
        "min_child_weight": (0, 5),
         "n_estimators": (50, 100),
        "scale_pos_weight": (1e-6, 500, "log-uniform"),
    })
# KNeighborsClassifier
models["model_name"].append("KNeighbors")
models["raw_estimator"].append(KNeighborsClassifier())
models["search_space"].append({
        "n_neighbors": (1, 30),
        "p": (1, 2)
    })
# SVC
models["model_name"].append("SCV")
models["raw_estimator"].append(SVC())
models["search_space"].append({
    #"C": Real(1e-6, 1e+6, prior='log-uniform'),
    "C": Real(1e-4, 5e+2, prior='log-uniform'),
    "gamma": Real(1e-6, 1e+1, prior='log-uniform'),
    "degree": (1, 8),
    "kernel": (['poly', 'rbf', 'linear']),
})
# Decision Tree
models["model_name"].append("Decision Tree")
models["raw_estimator"].append(DecisionTreeClassifier())
models["search_space"].append({
    "max_depth": (1, 50),
})
# Random Forest
models["model_name"].append("Random Forest")
models["raw_estimator"].append(RandomForestClassifier())
models["search_space"].append({
    "max_depth": (1, 50),
    "n_estimators": (1, 100),
})
# AdaBoost
models["model_name"].append("AdaBoost")
models["raw_estimator"].append(AdaBoostClassifier())
models["search_space"].append({
        "n_estimators": (1, 100)
    })
# QuadraticDiscriminantAnalysis
models["model_name"].append("QuadraticDiscriminantAnalysis")
models["raw_estimator"].append(QuadraticDiscriminantAnalysis())
models["search_space"].append({
        "priors": [None]
    })
# GaussianNB
models["model_name"].append("GaussianNB")
models["raw_estimator"].append(GaussianNB())
models["search_space"].append({
        "priors": [None]
        })
# Gaussian Process
models["model_name"].append("Gaussian Process")
models["raw_estimator"].append(GaussianProcessClassifier())
models["search_space"].append({
    "kernel": [None],
})
# MLP
models["model_name"].append("MLP")
models["raw_estimator"].append(MLPClassifier())
models["search_space"].append({
        "alpha": (1e-9, 1.0, "log-uniform"),
        "learning_rate": (["constant", "invscaling", "adaptive"]),
        "max_iter": [1000]
    })

# create df to save results
df_results = pd.DataFrame(columns=["model_name", "tunner", "accuracy"])

# loop trough models
for m in range(len(models["model_name"])):
    model_name = models["model_name"][m]
    if model_name in ["QuadraticDiscriminantAnalysis", "GaussianNB", "Gaussian Process"]:
        n_iter = 1
    else:
        n_iter = no_iter

    t = Tunner(
            models["raw_estimator"][m],
            models["search_space"][m],
            n_iter
            )
    result = t.opt.fit(X_train, y_train, callback=status_print)
    # save the results
    df_results = df_results.append({
            "model_name": model_name,
            "tunner": deepcopy(t.opt),
            "accuracy": np.round(t.opt.best_score_, 4)
            }, ignore_index=True)
