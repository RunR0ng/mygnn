from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.explain import GNNExplainer
from createDataset import NSSADataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch_geometric.datasets import TUDataset


# same as GNN.py
hidden_channels=64
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = gap(x, batch)
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.lin(x)
        return x


transform = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
dataset = NSSADataset(root='data/', transform=transform)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = dataset[:100]
test_dataset = dataset[100:]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader= DataLoader(test_dataset, batch_size=32, shuffle=True)

torch.set_default_tensor_type(torch.DoubleTensor)
num_classes = 5
model = GCN(dataset.num_features, num_classes)
for name,param in model.named_parameters():
    print(name,":",param.shape)

criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer=torch.optim.SGD(model.parameters(), lr=1e-3)
for param_group in optimizer.param_groups:
    print(param_group.keys())
    print([type(value) for value in param_group.values()])

def train():
    model.train()
    loss_all = 0
    # train_dataset
    for data in train_loader:
        # data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss_all += data.num_graphs * loss.item()
        loss.backward()
        optimizer.step()
    return loss_all / len(train_loader.dataset)

def test(loader):
    model.eval()

    correct = 0
    with torch.no_grad():
        for data in loader:  # Iterate in batches over the training/test dataset.
            # data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1,100):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if (epoch % 10 == 0):
        print(f'Epoch {epoch:3d}, loss={loss:.4f}')
        print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')