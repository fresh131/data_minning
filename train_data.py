import os
import pandas as pd
from functools import reduce
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
'''
脚本结构：
         read_and_preprocess_data(file_path)，         功能： 第一步中位数填补，第二步z分数标准化，第三步标准化后的0填补
         correlation_caculate(feature_label, method)， 功能： 计算三种相关系数，根据相关系数对特征排序
         rank_weight(n, methods = 'linear')，          功能： 设计了一个根据排名的权重函数，越靠前权重越高
         select_final_features(merged_result, x=64)，  功能： 根据三种相关系数计算联合权重，并进行特征综合筛选
         main(num_of_selected_features),               功能： 主框架，用来调用上述函数

脚本功能：
   对数据进行处理筛选出相关性高的前num_of_selected_features个特征，该值可于main中修改，得到的数据用于后续训练模型。
'''

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

def correlation_caculate(feature_label, method):
   '''
   这个函数用来计算相关系数
   :param feature_label: 只有特征值和标签的数据
   :param method:  要使用的相关系数计算方法
   :return: 返回相关系数从大到小的序列以及他们的排名
   '''
   correlation_pearson = feature_label.corr(method)
   correlation_with_label = abs(correlation_pearson['label'])
   sorted_correlation = correlation_with_label.drop('label').sort_values(ascending=False)
   selected_features = sorted_correlation.index[:107]

   result = {}
   rank = {}

   for i, feature in enumerate (selected_features,start=1):
       correlation_value = correlation_pearson.loc[feature, 'label']
       result[feature] = correlation_value
       rank[feature] = i

   result_df = pd.DataFrame(list(result.items()), columns=['Feature', f'{method} Correlation'])
   rank_df = pd.DataFrame(list(rank.items()), columns=['Feature', f'{method} Correlation Rank'])

   #保存到correlation目录下
   # with open('correlation/'+method+'_correlation.txt', 'w') as file:
   #     for feature in selected_features:
   #         correlation_value = correlation_pearson.loc[feature, 'label']
   #         file.write(f"{feature}: {correlation_value}\n")
   # print(method + 'correlation has caculated and saved')
   print(method + 'correlation has caculated')
   return result_df,rank_df


def rank_weight(n, methods = 'linear'):
    '''
    :param n: 在相关系数中的位次
    :param methods:
           linear 根据线性递减，位次越高权重越高
           mean   不设权重
    :return:
    '''
    if methods =='linear':
        return  1-(n-1)/106  #根据排名来计算权重
    elif methods == 'mean':
        return 1             #直接相加
    else:
        raise ValueError("参数'methods'输入有误. 请使用 'linear' or 'mean'.")


def select_final_features(merged_result, x=64):
    '''
    该函数旨在计算三种相关系数的联合系数，并进行排序，从而选出前x个相关性最高的特征
    :param merged_result:   其中的排列为 feature ,pearsonCorr, pearsonRank, spearmanCorr, spearmanRank, kendallCorr, kendallRank
    :x : 提取多少个
    :return: 返回一个根据综合相关系数得到的综合排序，以对特征进行一个筛选。并保存筛出出的特征名称，以在有新数据来到时直接预处理后提取。
    '''
    merged_result['weighted_absolute_sum'] = merged_result.apply(
        lambda row: sum([abs(row['pearson Correlation']) * rank_weight(row['pearson Correlation Rank']),
                         abs(row['spearman Correlation']) * rank_weight(row['spearman Correlation Rank']),
                         abs(row['kendall Correlation']) * rank_weight(row['kendall Correlation Rank'])]),
        axis=1
    )

    #排序
    merged_result_sorted = merged_result.sort_values(by='weighted_absolute_sum', ascending=False)

    selected_feature = merged_result_sorted['Feature'].head(x).tolist()
    feature_combined_corr = merged_result_sorted['weighted_absolute_sum'].head(64).tolist()

    return selected_feature, feature_combined_corr

def corr_select(num_of_selected_features):
    # 1. 读取数据，将原数据中的缺失值填补为中位数
    path = 'data/train_10000.csv'  # 验证集路径：'data/validate_1000'

    feature_label = read_and_preprocess_data(path)

    pearsonCorr, pearsonRank = correlation_caculate(feature_label,'pearson')
    spearmanCorr, spearmanRank = correlation_caculate(feature_label, 'spearman')
    kendallCorr, kendallRank = correlation_caculate(feature_label, 'kendall')

    # 先合并为一个，再使用 reduce 函数逐个合并 DataFrame，根据 'Feature' 列合并
    correlation_dfs = [pearsonCorr, pearsonRank, spearmanCorr, spearmanRank, kendallCorr, kendallRank]
    merged_result = reduce(lambda left, right: pd.merge(left, right, on='Feature'), correlation_dfs)

    #导出一个表格
    #merged_result.to_csv('data/merged_result_of_threeCorr_.csv', index=False)

    selected_feature, feature_combined_corr = select_final_features(merged_result,num_of_selected_features)

    #将selected_feature, feature_combined_corr组合一下保存一下
    # result_of_processed_data = pd.DataFrame({
    #     'Feature': selected_feature,
    #     'Combined Correlation': feature_combined_corr
    # })
    with open('selected_feature.txt', 'w') as file:
        for feature in selected_feature:
            file.write(f"{feature}\n")
    # # 将 result_of_processed_data 保存为 CSV 文件
    # result_of_processed_data.to_csv('data/result_of_processed_data.csv', index=False)

    selected_data = feature_label.loc[:, selected_feature + ['label']]
    selected_data.to_csv('data/corr/selected_train_data_'+str(num_of_selected_features)+'Features.csv', index=False)
    print(selected_data)

def rfe_select(num_of_selected_features = 64):

    path = 'data/train_10000.csv'  # 验证集路径：'data/validate_1000'
    feature_label = read_and_preprocess_data(path)

    features = feature_label.drop('label',axis=1)
    label = feature_label['label']


    # clf = SVC(decision_function_shape='ovr',kernel='linear')
    # rfe = RFE(clf, n_features_to_select=num_of_selected_features,importance_getter='coef_')

    # clf = LogisticRegression(multi_class='ovr')
    # rfe = RFE(clf, n_features_to_select=num_of_selected_features)

    clf = RandomForestClassifier()
    rfe = RFE(clf,n_features_to_select=num_of_selected_features)
    rfe.fit(features,label)

    select_features = features.columns[rfe.support_].tolist()
    # 从原始数据中提取选中的特征列和标签列
    selected_data = feature_label.loc[:, select_features + ['label']]

    # 存入CSV文件
    selected_data.to_csv('data/rfe/selected_train_data_'+str(num_of_selected_features)+'features_Logicrfe.csv', index=False)

    #存入txt文件
    with open('selected_feature_rfe.txt','w+') as file:
        for index in select_features:
            file.write(index + '\n') #f"{feature}\n"
    print(selected_data)

def main(pattern = 'corr', num_of_selected_features=64):

    if pattern == 'corr':
        corr_select(num_of_selected_features)
    elif pattern == 'rfe':
        rfe_select(num_of_selected_features)
    else:
        raise ValueError("参数'pattern'输入有误. 请使用 'corr' or 'rfe'.")


if __name__ =='__main__':

    pattern = 'corr'
    num=107
    main(pattern,num)

