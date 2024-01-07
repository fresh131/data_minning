from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import _joblib as jb
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.tree import export_graphviz
from io import StringIO
import pydotplus

df = pd.read_csv('data/corr/selected_train_data_107features.csv')

# npy数据训练
#df = np.load('data/train.npy')
# x_train = df[:, 0:-1]  # 选择所有特征列
# y_train = df[:, -1]    # 选择最后一列作为标签列


# 提取特征（Xtrain）和标签（Ytrain）
x_train = df.iloc[:, 0:-1]  # 选择所有特征列
y_train = df.iloc[:,   -1]    # 选择最后一列作为标签列

#随机森林
rf = RandomForestClassifier(criterion='entropy', n_estimators=75,
      max_depth=None,  # 定义树的深度，可以用来防止过拟合
      random_state=32
    # min_samples_split=10,  # 定义至少多少个样本的情况下才继续分叉
    # min_samples_leaf=0.02  # 定义叶子节点最少需要包含多少个样本（百分比表达），防止过拟合   名称带P则不带这辆个参数
)

#SVM
# rf = SVC(decision_function_shape='ovr',kernel='linear')

#逻辑回归
# rf = LogisticRegression(multi_class='ovr')

# 模型训练
rf.fit(x_train, y_train)

# modelname = 'Logic_107Features.pkl'
# modelname = 'SVM_std_107Features.pkl'
modelname = 'random_forest_model_n75p_std_107Features.pkl'


jb.dump(rf,'models/corr/'+modelname)
# jb.dump(rf,'models/rfe/'+modelname)

print('训练结束，模型保存为:'+modelname)



#保存决策树图像
# Estimators = rf.estimators_
# class_names = ['0', '1', '2', '3', '4', '5']
# feature_names = df.columns[1:-1]

# for index, model in enumerate(Estimators):
#     dot_data = StringIO()
#     export_graphviz(model, out_file=dot_data,
#                     feature_names=feature_names,
#                     class_names=class_names,
#                     filled=True, rounded=True,
#                     special_characters=True)
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#     graph.write_png('pic/n100/Rf{}.png'.format(index))

    # plt.figure(figsize=(20, 20))
    # plt.imshow(plt.imread('pic/Rf.png'.format(index)))
    # plt.axis('off')
    # plt.show(block=False)
    #
    # while plt.fignum_exists(index + 1):
    #     plt.pause(1)
    #
    # plt.close()