path = r'C:\Users\28929\Desktop\data.csv'
path_shell = r'C:\Users\28929\Desktop\data_shell.csv'
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
# 读取数据
df = pd.read_csv(path_shell)
# 将数据划分为特征和标签
X = df.iloc[:, :-3]
y = df.iloc[:, -1:]
#print(y)
# 数据归一化
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=46)
# 定义模型
model = MLPClassifier(hidden_layer_sizes=(32, 32), activation='relu', solver='lbfgs', max_iter=200)
# 训练模型
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)
print(y_pred)
# 打印模型的准确率
print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))