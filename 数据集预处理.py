import numpy as np
import pandas as pd
import torch

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, LabelBinarizer, Normalizer
from sklearn.preprocessing import MinMaxScaler

import time


def timeStampToDate(timeStamp):
    timeArray = time.localtime(timeStamp)
    return time.strftime("%Y-%m-%d %H:%M:%S", timeArray)


# 加载数据
df = pd.read_csv('./unsw/UNSW-NB15.csv', delimiter=",")
# 流量数据
df['ltime'] = pd.to_datetime(df['ltime'].map(timeStampToDate))
df['attack_cat'] = df['attack_cat'].str.strip()
df.dropna(subset=['attack_cat'], inplace=True)
attack_feature = pd.get_dummies(df['attack_cat'])
# attack_feature.drop(axis=1, columns=['Normal'])

df = df[['ltime', 'srcip', 'dstip', 'sbytes', 'dbytes', 'spkts',
         'dpkts', 'sloss', 'dloss', 'sload', 'dload']]
df = pd.concat([df, attack_feature], axis=1)
print(df.head())
print(df.shape)
df.to_csv('UNSW-NB15-1.csv',index=None)

# g = df.groupby(pd.Grouper(freq='2T', key='ltime')).sum(numeric_only=True)
# 按照delta_T= 2min聚合数据
g2 = df.groupby(pd.Grouper(key='ltime', freq='2T')).mean(numeric_only=True)
# g2.sort_values('stime', ascending=False)
# g2.drop(axis=1, columns=['id'], inplace=True)
g2.dropna(axis=0, how='any', inplace=True)
print(g2.shape)
# g2.to_csv('./unsw/g2.csv')

X = MinMaxScaler().fit_transform(g2)
X = pd.DataFrame(X)
print(X.head())
X.to_csv('./unsw/all_X.csv', index=False, header=0)
