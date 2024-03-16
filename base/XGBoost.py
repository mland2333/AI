path = r'C:\Users\28929\Desktop\data.csv'
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
# 读取数据
df = pd.read_csv(path)

# 将数据划分为特征和标签
X = df.iloc[:, :-3]
y = df.iloc[:,-3:]
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)
# 数据归一化
#print(y)
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X_pca)
#y_encoded = to_categorical(y, num_classes=2)
#y_decimal = np.sum(y * 2 ** np.arange(y.shape[1] - 1, -1, -1), axis=1)
#y_encoded = to_categorical(y_decimal, num_classes=8)
#print(y_encoded)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=70)
Nn = 20
Nm = 10
# 定义模型
x_coor = [0 for index in range(Nn*Nm)]
y_coor = [0 for index in range(Nn*Nm)]
z_coor = [0 for index in range(Nn*Nm)]

n=0
m=0
s=0
max_score=0
for i in range(Nn):
    for j in range(Nm):
        x_coor[i*Nm+j] = (i+1)*5
        y_coor[i*Nm+j] = 2+j
        #model = RandomForestClassifier(n_estimators=(i+1)*5, max_depth=2+j, random_state=46)
        model = xgb.XGBClassifier(n_estimators=(i+1)*5, max_depth=2+j, random_state=46,learning_rate=0.1
                          ,subsample=0.4,colsample_bytree=0.5)
        model.fit(X_train, y_train)
        z_coor[i*Nm+j] = model.score(X_test, y_test)
        if model.score(X_test, y_test) > max_score:
            n,m,s=(i+1)*5,5+j,model.score(X_test, y_test)
print(z_coor)
print(n,m,s)
xi=np.linspace(min(x_coor),max(x_coor))
yi=np.linspace(min(y_coor),max(y_coor))
xi,yi=np.meshgrid(xi,yi)
zi=griddata(np.array([x_coor,y_coor]).T,z_coor,(xi,yi),method='cubic')


fig=plt.figure()
ax = plt.axes(projection='3d')

surf=ax.plot_surface(xi,yi,zi,cmap='BuPu',linewidth=0,antialiased=False)
fig.colorbar(surf)

ax.set_xlabel('n_estimators')
ax.set_ylabel('max_depth')
ax.set_zlabel('accuracy')
plt.show()
'''for i in range(N):
    model = xgb.XGBClassifier(n_estimators=50, max_depth=8, random_state=70,learning_rate=0.1
                          ,subsample=0.4,colsample_bytree=0.1*(i+1))
    model.fit(X_train, y_train)
    x_coor[i] = 0.1*(i+1)
    y_coor[i] = model.score(X_test, y_test)'''


'''plt.plot(x_coor,y_coor,'s-',color = 'r',label="test_accuracy")
plt.xlabel("colsample_bytree")#横坐标名字
plt.ylabel("accuracy")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()
'''

# 训练模型
'''model.fit(X_train, y_train)
pred = model.predict(X_test)
print(pred)
# 打印模型的准确率
print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))'''