from torch_geometric.nn import TopKPooling, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
from createDataset import NSSADataset


embed_dim = 128


class Net(torch.nn.Module):  # 针对图进行分类任务
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = SAGEConv(10, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=1, embedding_dim=128)

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 5)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # x:n*1,其中每个图里点的个数是不同的
        # print(x)
        # x = self.item_embedding(input=x)  # n*1*128 特征编码后的结果
        # print('item_embedding',x.shape)
        # x = x.squeeze(1)  # n*128
        # print('squeeze',x.shape)
        x = F.relu(self.conv1(x, edge_index))  # n*128
        # print('conv1',x.shape)

        x, edge_index, _, batch, _, _ = self.pool1(
            x, edge_index, None, batch)  # pool之后得到 n*0.8个点
        # print('self.pool1',x.shape)
        # print('self.pool1',edge_index)
        # print('self.pool1',batch)
        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x1 = gap(x, batch)
        # print('gmp',gmp(x, batch).shape) # batch*128
        # print('cat',x1.shape) # batch*256
        x = F.relu(self.conv2(x, edge_index))
        # print('conv2',x.shape)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # print('pool2',x.shape)
        # print('pool2',edge_index)
        # print('pool2',batch)
        # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x2 = gap(x, batch)
        # print('x2',x2.shape)
        x = F.relu(self.conv3(x, edge_index))
        # print('conv3',x.shape)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        # print('pool3',x.shape)
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x3 = gap(x, batch)
        # print('x3',x3.shape)# batch * 256
        x = x1 + x2 + x3  # 获取不同尺度的全局特征

        x = self.lin1(x)
        # print('lin1',x.shape)
        x = self.act1(x)
        x = self.lin2(x)
        # print('lin2',x.shape)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.lin3(x)
        # print('sigmoid',x.shape)
        return x


def train():
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data
        # print('data',data)
        optimizer.zero_grad()
        output = model(data)
        label = data.y
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_loader.dataset)


transform = T.Compose(
    [T.NormalizeFeatures(), T.RemoveIsolatedNodes(), T.ToUndirected()])
dataset = NSSADataset(root='data1/', transform=transform)
train_loader = DataLoader(dataset, batch_size=32)

torch.set_default_tensor_type(torch.FloatTensor)
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
crit = torch.nn.CrossEntropyLoss()


def test(loader):
    model.eval()

    correct = 0
    with torch.no_grad():
        # Iterate in batches over the training/test dataset.
        for data in loader:
            # data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            # Check against ground-truth labels.
            correct += int((pred == data.y).sum())
    # Derive ratio of correct predictions.
    return correct / len(loader.dataset)


for epoch in range(100):
    train_acc = test(train_loader)
    loss = train()
    if (epoch % 10 == 0):
        print(f'Epoch {epoch:3d}, loss={loss:.4f}')
        print(f'Train Acc: {train_acc:.4f}')

# def evalute(loader, model):
#     model.eval()

#     prediction = []
#     labels = []

#     with torch.no_grad():
#         for data in loader:
#             data = data  # .to(device)
#             pred = model(data)  # .detach().cpu().numpy()

#             label = data.y  # .detach().cpu().numpy()
#             prediction.append(pred)
#             labels.append(label)
#     prediction = np.hstack(prediction)
#     labels = np.hstack(labels)

#     return roc_auc_score(labels, prediction)


# for epoch in range(1):
#     roc_auc_score = evalute(dataset, model)
#     print('roc_auc_score', roc_auc_score)
