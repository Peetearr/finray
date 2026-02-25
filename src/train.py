import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter('runs/experiment')

df = pd.read_csv('data/finray.csv')
X = df[['p1','p2','p3','p4','p5','p6','p7']].values.astype('float32')
y = df[['y_point','force']].values.astype('float32')

scaler_X, scaler_y = StandardScaler(), StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
train_size = int(0.9 * len(dataset))
train_data, test_data = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

model = nn.Sequential(
    nn.Linear(7, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

steps = []
losses = []

for epoch in range(5000):
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch % 20 == 0:
        losses.append(loss.item())
        steps.append(epoch)
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    writer.add_scalar('Loss/train', loss.item(), epoch)
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

data = np.array([steps, losses]).T
np.savetxt('training.csv', data, delimiter=',', header='Эпохи, Потери', comments='')

# plt.plot(steps, losses)
# plt.show()

model.eval()
with torch.no_grad():
    test_loss = 0
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        test_loss += criterion(pred, yb).item()
    writer.add_scalar('Loss/test', test_loss/len(test_loader), epoch)
    print(f'Test Loss: {test_loss/len(test_loader):.4f}')
writer.close()