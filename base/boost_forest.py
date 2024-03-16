import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import xgboost as xgb

path = r'C:\Users\28929\Desktop\data.csv'# 读取数据

df = pd.read_csv(path, header=0)
X = df.iloc[:, :-3]
y = df.iloc[:, -3:-2]# 将数据划分为特征和标签
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X)
plt.bar(range(6), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.xticks(range(6), ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()# 特征工程
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)# 数据归一化
#mlb = MultiLabelBinarizer()
#y_encoded = mlb.fit_transform(y.values)# 处理标签数据
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=46)
model = xgb.XGBClassifier(random_state=50)# 定义模型
# 超参数调整
param_grid = {'max_depth': [3, 5, 7,8], 'learning_rate': [0.1,0.2, 0.3, 0.5], 'n_estimators': [13,15, 19, 10]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
model_best = grid_search.best_estimator_
model_best.fit(X_train, y_train)# 训练模型
y_pred_train = model_best.predict(X_train)
y_pred_test = model_best.predict(X_test)
print(y_pred_test)
print("Train Accuracy:", classification_report(y_train, y_pred_train))
print("Test Accuracy:", classification_report(y_test, y_pred_test))# 打印模型的评估指标
'''y_pred_proba = model_best.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred_proba.ravel())
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()# 模型评估'''
