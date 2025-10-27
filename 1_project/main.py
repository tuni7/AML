import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from scipy.stats import chi2

# For each NaN entry insert a random entry of the same feature. 
def impute(X):
    for j in range (len(X[0])):
        for i in range (len(X)):
            ran = np.nan
            if (np.isnan(X[i, j])):
                while (np.isnan(ran)):
                    ran = random.choice(X[:, j])
                X[i, j] = ran
    return X

#Erase equal entries (features)
def errase_equal_entries(X):
    vec = []
    for j in range (len(X[0])):
        val = X[0, j]
        for i in range (len(X)):
            if X[i, j] != val:
                break
            if i == len(X) - 1:
                vec.append(j)
    return np.delete(X, vec, 1)

# Use Mahalanobis distance to detect outliers. (X - mu)* Covariance_matrix^-1*(X - mu)^T
def filter(X, y):
    _mean = np.mean(X, axis=0, keepdims=False)
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    matrix = np.matmul(inv_cov, np.matrix.transpose(X - _mean))
    matrix = np.matmul((X - _mean), matrix)
    D2 = np.diag(matrix)
    return X[D2 < (chi2.ppf(0.995, df=len(X[0])))], y[D2 < (chi2.ppf(0.995, df=len(X[0])))]


_X_train = pd.read_csv("data/X_train.csv")
_X_test = pd.read_csv("data/X_test.csv")
_y_train = pd.read_csv("data/y_train.csv")

X_train = _X_train.values[:, 1:]
y_train = _y_train.values[:, 1:]
X_test = _X_test.values[:, 1:]

X_test = impute(X_test)
X_train = impute(X_train)

X_train = errase_equal_entries(X_train)

X_train, y_train = filter(X_train, y_train)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

