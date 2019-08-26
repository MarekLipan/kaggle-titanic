"""
Kaggle Titanic Challenge
"""

import pandas as pd
import numpy as np

# load the datasets
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# transforming the data
cols_X = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
col_y = ["Survived"]
train_X = train.loc[:, cols_X]
train_y = train.loc[:, col_y]
# Sex (=1 if male)
train_X["Male"] = (train_X["Sex"]=="male")*1
# dummy variables for point of departure (if all 0 then it is nan)
train_X["Embarked_S"] = (train_X["Embarked"]=="S")*1
train_X["Embarked_C"] = (train_X["Embarked"]=="C")*1
train_X["Embarked_Q"] = (train_X["Embarked"]=="Q")*1
# drop the encoded cols
train_X = train_X.drop(["Sex", "Embarked"], axis=1)
# replace age nans with -1 and let the model know
train_X["Age_nan"] = (np.isnan(train_X["Age"]))*1
train_X["Age"][np.isnan(train_X["Age"])]=-1
