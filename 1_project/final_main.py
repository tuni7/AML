import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from scipy.stats import chi2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.utils.data as data_utils
import time
# for features selection:
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from lightgbm import LGBMRegressor



class Mri_dataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        data = scaler.transform(X)
        self.X = torch.tensor(data, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_dim_1 = 600
        hidden_dim_2 = 100
        self.layer1 = nn.Linear(input_dim, hidden_dim_1)
        self.layer2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.layer3 = nn.Linear(hidden_dim_2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        return self.layer3(x)
    



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
def erase_equal_entries(X, X_test):
    vec = []
    for j in range (len(X[0])):
        val = X[0, j]
        for i in range (len(X)):
            if X[i, j] != val:
                break
            if i == len(X) - 1:
                vec.append(j)
    return np.delete(X, vec, 1), np.delete(X_test, vec, 1)

# Use Mahalanobis distance to detect outliers. (X - mu)* Covariance_matrix^-1*(X - mu)^T
def filter(X, y):
    _mean = np.mean(X, axis=0, keepdims=False)
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    matrix = np.matmul(inv_cov, np.matrix.transpose(X - _mean))
    matrix = np.matmul((X - _mean), matrix)
    D2 = np.diag(matrix)
    return X[D2 < (chi2.ppf(0.995, df=len(X[0])))], y[D2 < (chi2.ppf(0.995, df=len(X[0])))]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Data preperation

_X_train = pd.read_csv("data/X_train.csv")
_X_test = pd.read_csv("data/X_test.csv")
_y_train = pd.read_csv("data/y_train.csv")

X_train = _X_train.values[:, 1:]
y_train = _y_train.values[:, 1:]
X_test = _X_test.values[:, 1:]

X_test = impute(X_test)
X_train = impute(X_train)

X_train, X_test = erase_equal_entries(X_train, X_test)

X_train, y_train = filter(X_train, y_train)

# X_train = SelectKBest(f_regression, k=700).fit_transform(X_train,y_train[:,0])
# X_train = SelectKBest(mutual_info_regression, k=650).fit_transform(X_train,y_train[:,0])
selector1 = SelectKBest(f_regression, k=700)
X_train = selector1.fit_transform(X_train, y_train[:, 0])
X_test = selector1.transform(X_test)

selector2 = SelectKBest(mutual_info_regression, k=650)
X_train = selector2.fit_transform(X_train, y_train[:, 0])
X_test = selector2.transform(X_test)

# train_data = Mri_dataset(X_train, y_train)
# val_data = Mri_dataset(X_val, y_val)
# train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=16, shuffle=True)

# num_features = len(X_train[0])

# Training process

# kernel = 1.0 * RBF(1.0)
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8
)
train_with_val = False #set to false if you want to submit a solution!!
if train_with_val:
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = r2_score(y_val, y_pred)
    print(f"R2 score for validation set is: {round(score, 2)}")
else:
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    model.fit(X_train, y_train)


X_test = scaler.transform(X_test)
pred = model.predict(X_test)
ids = _X_test["id"]
submission = pd.DataFrame({
    "id": ids,
    "target": pred
})
submission.to_csv("submission.csv", index=False)

# gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(X_train, y_train)
# y_pred = gpc.predict(X_val)
# score = r2_score(y_val, y_pred)
# print(r2_score(y_train, gpc.predict(X_train)))
# print(f"R2 score for validation set is: {round(score, 2)}")
# model = Model(num_features).to(device)
# # model = nn.Linear(num_features, 1).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# epochs = 10

# for k in range(epochs):
#     train_loss = 0
#     val_loss = 0
#     model.train()
#     for x_batch, y_batch in train_loader:
#         x_batch = x_batch.to(device)
#         y_batch = y_batch.to(device)
#         pred = model(x_batch)
#         loss = criterion(pred, y_batch)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#     print(f"Epoch {k + 1}: train_loss = {round(train_loss/len(X_train), 2)}")
#     model.eval()
#     with torch.no_grad():
#         for x_batch, y_batch in val_loader:
#             x_batch = x_batch.to(device)
#             y_batch = y_batch.to(device)
#             pred = model(x_batch)
#             val_loss += criterion(pred, y_batch).item()
#         print(f"Validation loss: {round(val_loss/len(X_val), 2)}")
    




#Calculate score
# model.eval()
# with torch.no_grad():
#     preds = []
#     trues = []
#     for x_batch, y_batch in val_loader:
#         x_batch = x_batch.to(device)
#         preds.append(model(x_batch).squeeze(-1).cpu())
#         trues.append(y_batch.squeeze(-1).cpu())

# y_pred = torch.cat(preds).numpy()
# y_true = torch.cat(trues).numpy()
# score = r2_score(y_true, y_pred)
# print(f"R2 score for validation set is: {round(score, 2)}")

