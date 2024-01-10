import os
import pandas as pd
from functools import reduce


# 读取数据
df = pd.read_csv('data/test_2000_x.csv')
features = df.iloc[:, 1:]
# 使用每列的中位数填充缺失值
features = features.fillna(features.median())
# 使用z分数标准化，并将标准化后空缺值补为0
numeric_features = features.dtypes[features.dtypes != 'object'].index
features[numeric_features] = features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
features[numeric_features] = features[numeric_features].fillna(0)

with open('selected_RFfeature_rfe.txt', 'r') as file:
    selecter_feature_names = file.read().splitlines()

selected_features = features[selecter_feature_names]
selected_features.to_csv('data/el/RandomForest_test_64Features.csv',index=False)
print(selected_features)

