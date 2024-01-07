
import pandas as pd
from sklearn.utils import _joblib as jb
from sklearn.metrics import accuracy_score
import numpy as np

# 模型路径
model_path1 = 'models/corr/random_forest_model_n75p_std_107Features.pkl'
model_path2 = 'models/corr/SVM_107Features.pkl'
model_path3 = 'models/corr/Logic_107Features.pkl'
# model_path4 = 'models/corr/random_forest_model_n150P_std_64Features.pkl'
# model_path5 = 'models/corr/random_forest_model_n200P_std_64Features.pkl'

# 加载模型
rf_model1 = jb.load(model_path1)
rf_model2 = jb.load(model_path2)
rf_model3 = jb.load(model_path3)

# rf_model4 = jb.load(model_path4)
# rf_model5 = jb.load(model_path5)

# 验证数据路径
validate_data_path1 = 'data/corr/selected_validate_data_107Features.csv'
validate_data_path2 = 'data/corr/selected_validate_data_107Features.csv'
validate_data_path3 = 'data/corr/selected_validate_data_107Features.csv'
# validate_data_path4 = 'data/rfe/selected_validate_data_64Features_Logicrfe.csv'
# validate_data_path5 = 'data/rfe/selected_validate_data_64Features_SVMrfe.csv'

# validate_data_path6 = 'data/corr/selected_validate_data_64Features.csv'
# 读取验证数据
df_validate1 = pd.read_csv(validate_data_path1)
df_validate2 = pd.read_csv(validate_data_path2)
df_validate3 = pd.read_csv(validate_data_path3)
# df_validate4 = pd.read_csv(validate_data_path6)
# df_validate5 = pd.read_csv(validate_data_path6)


# 提取特征和标签
x_validate1, y_validate1 = df_validate1.iloc[:, 0:-1], df_validate1.iloc[:, -1]
x_validate2, y_validate2 = df_validate2.iloc[:, 0:-1], df_validate2.iloc[:, -1]
x_validate3, y_validate3 = df_validate3.iloc[:, 0:-1], df_validate3.iloc[:, -1]
# x_validate4, y_validate4 = df_validate4.iloc[:, 0:-1], df_validate4.iloc[:, -1]
# x_validate5, y_validate5 = df_validate5.iloc[:, 0:-1], df_validate5.iloc[:, -1]

# 预测
y_pred1 = rf_model1.predict(x_validate1)
y_pred2 = rf_model2.predict(x_validate2)
y_pred3 = rf_model3.predict(x_validate3)
# y_pred4 = rf_model4.predict(x_validate4)
# y_pred5 = rf_model5.predict(x_validate5)


# 准确度
accuracy1 = accuracy_score(y_validate1, y_pred1)
print(f'RF Accuracy: {accuracy1}')

accuracy2 = accuracy_score(y_validate2, y_pred2)
print(f'SVM Accuracy: {accuracy2}')

accuracy3 = accuracy_score(y_validate3, y_pred3)
print(f'Logic Accuracy: {accuracy3}')

# accuracy4 = accuracy_score(y_validate4, y_pred4)
# print(f'random_forest_model_n150P Accuracy: {accuracy4}')
#
# accuracy5 = accuracy_score(y_validate5, y_pred5)
# print(f'random_forest_model_n200P Accuracy: {accuracy5}')





# import pandas as pd
# from sklearn.model_selection import train_test_split
# from matplotlib import pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import RFECV
#
# # 读取CSV文件
# df = pd.read_csv('train_10000.csv')
#
# features = df.iloc[:, 1:-1]
# numeric_features = features.dtypes[features.dtypes != 'object'].index
# features[numeric_features] = features[numeric_features].apply(
#     lambda x: (x - x.mean()) / (x.std())
# )
# # 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
# features[numeric_features] = features[numeric_features].fillna(0)
# features_labels = pd.concat([features, df[['label']]], axis=1)
# #划分出训练数据和训练标签
# train_features = pd.concat([df[['sample_id']], features], axis=1)
# train_label = df[['sample_id', 'label']]
#
# df = pd.concat([train_features, train_label[['label']]], axis=1)
#
#
#
# # import numpy as np
# # from sklearn.datasets import load_breast_cancer
# # from sklearn.model_selection import train_test_split
# # from sklearn.naive_bayes import GaussianNB
# # from sklearn.metrics import confusion_matrix
# # from matplotlib import pyplot as plt
# # import seaborn as sns
# #
# #
# # # 训练模型函数
# # def model_fit(x_train, y_train, x_test, y_test):
# #     clf = GaussianNB()
# #     clf.fit(x_train, y_train)  # 对训练集进行拟合
# #     print('accuracy on traning set:' + str(clf.score(x_train, y_train)))
# #     print('accuracy on test set:' + str(clf.score(x_test, y_test)))
# #     pred = clf.predict(x_test)
# #     cm = confusion_matrix(pred, y_test)
# #     return cm
# #
# # if __name__ == '__main__':
# #     cancer = load_breast_cancer()
# #     x, y = cancer.data, cancer.target
# #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=3)
# #     cm = model_fit(x_train, y_train, x_test, y_test)  #朴素贝叶斯
# #     # matplotlib_show(cm)
#
#
# # from sklearn.datasets import load_breast_cancer
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.model_selection import train_test_split
# # from sklearn import datasets
# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# # cancer = datasets.load_breast_cancer()
# # X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.42, stratify=cancer.target, random_state=42)
# # tree = DecisionTreeClassifier(random_state=0)
# #
# # tree.fit(X_train, y_train)
# # print("Accuracy on traning set:{:.3f}".format(tree.score(X_train, y_train)))
# # print("Accuracy on test set:{:.3f}".format(tree.score(X_test, y_test)))
# # print("tree max depth:{}".format(tree.tree_.max_depth)) #决策树
#
#
# # from sklearn import datasets
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.svm import SVC
# # from sklearn.metrics import accuracy_score, confusion_matrix
# #
# # diabetes = datasets.load_diabetes()
# # X = diabetes.data
# # Y = [1 if y>100 else 0 for y in diabetes.target]
# #
# # # 将数据集划分为训练集和测试集
# # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# #
# # # 数据标准化
# # scaler = StandardScaler()
# # X_train = scaler.fit_transform(X_train)
# # X_test = scaler.transform(X_test)
# #
# # # 创建 SVM 模型
# # svm_model = SVC(kernel='poly', class_weight='balanced')
# #
# # # 训练模型
# # svm_model.fit(X_train, y_train)
# #
# # # 输出准确度  SVM
# # print('Accuracy on training set:', svm_model.score(X_train,y_train))
# # print('Accuracy on test set:', svm_model.score(X_test,y_test))
#
# from sklearn.datasets import load_diabetes
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
#
# diabetes = load_diabetes()
# X, y = diabetes.data, diabetes.target
#
# threshold = 100  # 阈值
# y_binary = (y > threshold).astype(int)
# X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)
#
# rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
# rf_classifier.fit(X_train, y_train)
#
# print('Accuracy on training set:', rf_classifier.score(X_train,y_train))
# print('Accuracy on test set:', rf_classifier.score(X_test,y_test))
#
#
#
#
#
#
# # RFE
# # X = df.iloc[:, 1:-1]  # 选择从第2列到倒数第2列的所有特征列
# # y = df.iloc[:, -1]    # 选择最后一列作为标签列
#
# # # 创建随机森林分类器
# # clf = RandomForestClassifier(n_estimators=100, random_state=42)
# #
# # # 创建RFECV递归特征消除器，使用5折交叉验证
# # rfecv = RFECV(estimator=clf, step=1, cv=5, scoring='accuracy')
# #
# # # 运行RFECV递归特征消除器，并返回选择的特征
# # selected_features = rfecv.fit_transform(X, y)
# #
# # # 输出选择的特征数量和选择的特征的索引
# # print("Selected Features: %d" % rfecv.n_features_)
# # print("Feature Ranking: %s" % rfecv.ranking_)
