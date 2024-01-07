import os
import pandas as pd
from functools import reduce

def read_and_preprocess_data(file_path):
    '''
    对数据进行预处理： 1.  中位数填补
                    2.  z分数标准化
                    3.  0填补
                    4   去除第一列编号
    :param file_path: 文件路径
    :return: 特征和标签的组合
    '''
    # 读取数据
    df = pd.read_csv(file_path)
    features = df.iloc[:, 1:-1]
    labels = df[['label']]

    # 使用每列的中位数填充缺失值
    features = features.fillna(features.median())

    # 使用z分数标准化，并将标准化后空缺值补为0
    numeric_features = features.dtypes[features.dtypes != 'object'].index
    features[numeric_features] = features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    features[numeric_features] = features[numeric_features].fillna(0)

    # 将特征和标签合并
    feature_label = pd.concat([features, labels], axis=1)
    return feature_label


def corr_validate_data(num):
    path = 'data/validate_1000.csv'
    feature_label = read_and_preprocess_data(path)

    with open('selected_feature.txt','r') as file:
        selecter_feature_names = file.read().splitlines()

    selected_features = feature_label[selecter_feature_names]
    #selected_features = feature_label[selecter_feature_names[:32]]

    selected_validate_data = pd.concat([selected_features, feature_label['label']], axis=1)

    selected_validate_data.to_csv('data/corr/selected_validate_data_'+str(num)+'Features.csv',index=False)

    print('end')

def rfe_validate_data(num):
    path = 'data/validate_1000.csv'
    feature_label = read_and_preprocess_data(path)

    with open('selected_feature_rfe.txt','r') as file:
        selecter_feature_names = file.read().splitlines()

    selected_features = feature_label[selecter_feature_names]
    #selected_features = feature_label[selecter_feature_names[:32]]

    selected_validate_data = pd.concat([selected_features, feature_label['label']], axis=1)

    selected_validate_data.to_csv('data/rfe/selected_validate_data_'+str(num)+'Features_rfe.csv',index=False)

    print('end')


def main(pattern='corr',num=64):
    if pattern == 'corr':
        corr_validate_data(num)
    elif pattern == 'rfe':
        rfe_validate_data(num)
    else:
        raise ValueError("参数'pattern'输入有误. 请使用 'corr' or 'rfe'.")


if __name__ == '__main__':
    pattern = 'corr'
    num = 107
    main(pattern,num)