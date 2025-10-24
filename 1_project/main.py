import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random


def impute(X):
    for j in range (len(X[0])):
        for i in range (len(X)):
            ran = np.nan
            if (np.isnan(X[i, j])):
                while (np.isnan(ran)):
                    ran = random.choice(X[:, j])
                # print(f"j = {j} X_test = {X[i, j]} Randdom value = {ran}")
                X[i, j] = ran
    return X



_X_train = pd.read_csv("data/X_train.csv")
_X_test = pd.read_csv("data/X_test.csv")
_y_train = pd.read_csv("data/y_train.csv")


X_train = _X_train.values[:, 1:]
y_train = _y_train.values[:, 1:]
X_test = _X_test.values[:, 1:]


X_test = impute(X_test)
X_train = impute(X_train)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# for i in range (len(X_test)):
    # lin_without_nan = 
    # print(X_test[i, 0])
    # print(X_test[0])
# X_train = np.where(np.isnan(X_train), X_mean, X_train)
# X_val = np.where(np.isnan(X_val), X_mean, X_val)