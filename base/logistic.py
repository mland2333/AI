# 导入所需的库
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 读取数据集
df = pd.read_csv('data.csv')

# 分离输入和输出
X = df.iloc[:, :-2]
y = df.iloc[:, -1:]

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=46)

# 训练模型
lr = LogisticRegression()
lr.fit(X_train, y_train)

# 预测测试集
y_pred = lr.predict(X_test)

# 评估模型性能
print(classification_report(y_test, y_pred))
