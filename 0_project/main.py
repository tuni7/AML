import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


torch.manual_seed(42)
class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = 20
        self.layer1 = nn.Linear(4, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)




_train_data = pd.read_csv("train(2).csv")
_test_data = pd.read_csv("test(3).csv")
train_data = _train_data.values.astype(np.float32)
test_data = _test_data.values.astype(np.float32)

train_labels = train_data[:, -1]
train_data = np.delete(train_data, -1, axis=1)

N = int(len(train_data)*0.8)
train_x = torch.tensor(train_data[:N])
train_y = torch.tensor(train_labels[:N])
val_x = torch.tensor(train_data[N:])
val_y = torch.tensor(train_labels[N:])

test_x = torch.tensor(test_data)

model = Model()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 15



for k in range(epochs):
    epoch_loss = 0
    model.train()
    for i in range(len(train_x)):
        pred = model(train_x[i][1:5])
        loss = criterion(pred, train_y[i].squeeze().long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss
    correct = 0
    model.eval()
    print(f"Epoch: {k}: train_loss = {round(epoch_loss.item(), 2)}")
    for i in range(len(val_x)):
        pred = model(val_x[i][1:5])
        label = pred.argmax(dim=0)
        if (label == val_y[i].squeeze().long()):
            correct += 1
    print(f"Epoch: {k}: validation accuracy: {round(correct/len(val_y), 2)}")


model.eval()
ids = test_x[:, 0]
pred = model(test_x[:, 1:5])
label = pred.argmax(dim=1)

ids = _test_data["id"]
submission = pd.DataFrame({
    "id": ids,
    "target": label
})
submission.to_csv("submission.csv", index=False)