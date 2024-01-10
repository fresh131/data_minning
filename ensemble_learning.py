import os
import pandas as pd
from sklearn.utils import _joblib as jb
from scipy.stats import mode
import numpy as np

model_path1 = 'models/el/Logic_64Features.pkl'
model_path2 = 'models/el/RandomForest_64Features.pkl'
model_path3 = 'models/el/SVM_64Features.pkl'

data1 = pd.read_csv('data/el/Logic_test_64Features.csv')
data2 = pd.read_csv('data/el/RandomForest_test_64Features.csv')
data3 = pd.read_csv('data/el/SVM_test_64Features.csv')

model1 = jb.load(model_path1)
model2 = jb.load(model_path2)
model3 = jb.load(model_path3)

pred1 = model1.predict(data1)
pred2 = model2.predict(data2)
pred3 = model3.predict(data3)

pred4 = np.loadtxt("data/mlp.txt",dtype=int)
pred5 = np.loadtxt("data/cnn.txt",dtype=int)

final_pred = mode([pred1, pred2, pred3, pred4, pred5], axis=0).mode[0]

np.savetxt('data/final_pred.txt', final_pred, fmt='%d')
print(final_pred)
