import pandas as pd
from sklearn.utils import _joblib as jb
from sklearn.metrics import accuracy_score
import numpy as np

model_path = 'models/rfe/random_forest_model_n75P_std_64Features_rfe.pkl'
# model_path = 'models/rfe/SVM_std_64Features_rfe.pkl'
# model_path = 'models/rfe/Logic_std_64Features_rfe.pkl'


rf_model = jb.load(model_path)
validate_data_path = 'data/rfe/selected_validate_data_64Features_rfe.csv'
df_validate = pd.read_csv(validate_data_path)

x_validate = df_validate.iloc[:, 0:-1]
y_validate = df_validate.iloc[:, -1]

# validate_data_path = 'data/rfe/selected_validate_data_64Features_rfe.csv'
# datas = np.load(validate_data_path)
#
# x_validate = datas[:, 0:-1]
# y_validate = datas[:,   -1]

# print(x_validate)
# print(y_validate)

y_pred = rf_model.predict(x_validate)
num_of_label = np.bincount(y_pred.astype(int))

# 准确度
accuracy = accuracy_score(y_validate, y_pred)
print(f'Model Accuracy: {accuracy}')

for index, num in enumerate(num_of_label, start=1):
    print('预测结果:{}  个数：{}'.format(index-1, num))

