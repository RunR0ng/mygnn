import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import InMemoryDataset
from torch import Tensor
import tqdm
import torch.nn.functional as F

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, LabelBinarizer, Normalizer
from sklearn.preprocessing import MinMaxScaler
import time


def timeStampToDate(timeStamp):
    timeArray = time.localtime(timeStamp)
    return time.strftime("%Y-%m-%d %H:%M:%S", timeArray)


df = pd.read_csv('UNSW-NB15-1.csv', delimiter=",")
# df = df[['ltime', 'srcip', 'dstip', 'sbytes', 'dbytes', 'spkts',
#          'dpkts', 'sloss', 'dloss', 'sload', 'dload',
#          'Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers',
#          'Generic', 'Reconnaissance', 'Shellcode', 'Worms']]
# df['ltime'] = pd.to_datetime(df['ltime'].map(timeStampToDate))
# df['attack_cat'].fillna('Normal', inplace=True)
# df.dropna(subset=['attack_cat'], inplace=True)
# df['attack_cat'] = df['attack_cat'].str.strip()
# mapping = {'Normal': 0, 'Reconnaissance': 1, 'Analysis': 2, 'Generic': 3, 'Exploits': 4,
#            'Fuzzers': 5, 'DoS': 6, 'Backdoor': 7, 'Shellcode': 8, 'Worms': 9}
# df['attack_cat'] = df['attack_cat'].map(mapping)

# one_hot = pd.get_dummies(df['attack_cat'])
# one_hot.columns = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers',
#                    'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
# df = pd.concat([df, one_hot], axis=1)

# 替换IP为mapped_id
unique_node_id = pd.read_csv('./unsw/node_id.csv')
s_id = pd.merge(df['srcip'], unique_node_id, left_on='srcip',
                right_on='IP', how='left')['nodeID']
d_id = pd.merge(df['dstip'], unique_node_id, left_on='dstip',
                right_on='IP', how='left')['nodeID']
e = pd.concat([s_id, d_id], axis=1)
# print(df[['srcip','dstip']].shape)
# print(e.shape)
df['srcip'] = s_id
df['dstip'] = d_id
df = df.loc[((df.srcip != -1) & (df.dstip != -1)), :]
df = df.reset_index(drop=True)


class NSSADataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(NSSADataset, self).__init__(root, transform,
                                          pre_transform)  # transform就是数据增强，对每一个数据都执行
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):  # 检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件
        # 如有文件不存在，则调用download()方法执行原始文件下载
        return []

    @property
    # 检查self.processed_dir目录下是否存在self.processed_file_names属性方法返回的所有文件，没有就会走process
    def processed_file_names(self):
        return ['NSSA_2T.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []
        n = 0

        y = pd.read_csv('./unsw/all_Y.csv', header=None).values
        groups = df.groupby(pd.Grouper(freq='2T', key='ltime'))

        for group in groups:
            g = group[1]  # 取出value值
            # g = g[g['srcip'] != g['dstip']]
            e = g[['srcip', 'dstip']]
            # e = pd.concat([s_id, d_id], axis=1)
            # e.columns = ['sid', 'did']
            # 统计 sip->dip 之间边的条数
            counts = e.groupby(['srcip', 'dstip']).size(
            ).reset_index(name='counts')
            e_index = np.array(counts[['srcip', 'dstip']]).T
            e_index = torch.tensor(e_index)
            e_attr = counts['counts']
            e_attr = torch.tensor(e_attr)
            # 不同节点的流量特征
            d1 = g[['srcip', 'sbytes', 'spkts', 'sloss', 'sload']
                   ].groupby('srcip').sum(numeric_only=True)
            d2 = g[['dstip', 'dbytes', 'dpkts', 'dloss', 'dload']
                   ].groupby('dstip').sum(numeric_only=True)
            # d1.columns = ['bytes', 'pkts', 'loss', 'load']
            # d2.columns = ['bytes', 'pkts', 'loss', 'load']
            # d1 = d1.loc[:, ['bytes', 'pkts', 'loss', 'load']]
            # d2 = d2.loc[:, ['bytes', 'pkts', 'loss', 'load']]
            # flow = d1.add(d2, fill_value=0)
            flow = pd.concat([d1, d2], axis=1)
            # 攻击特征
            attack = g[['srcip', 'Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers',
                        'Generic', 'Reconnaissance', 'Shellcode', 'Worms']].groupby(
                'srcip').sum(numeric_only=True)
            # 资产特征
            asset = pd.DataFrame({'deviceType': [3, 3, 3, 2, 1]})
            x = pd.concat([asset, flow, attack], axis=1)
            x.fillna(0, inplace=True)

            x = torch.DoubleTensor(np.array(x))

            yi = torch.LongTensor(y[n])
            data = Data(x=x, edge_index=e_index, edge_attr=e_attr, y=yi)

            n += 1
            # print('data:', data)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


dataset = NSSADataset(root='data/')
# transform = T.Compose([T.NormalizeFeatures(), T.RemoveIsolatedNodes(), T.ToUndirected()])
# dataset = NSSADataset(root='data/', transform=transform)
# print("数据加载完成！")

# data=dataset[0]
# print(f'Number of nodes: {data.num_nodes}')
# print(f'Number of edges: {data.num_edges}')
# print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
# print(f'Has isolated nodes: {data.has_isolated_nodes()}')
# print(f'Has self-loops: {data.has_self_loops()}')
# print(f'Is undirected: {data.is_undirected()}')
