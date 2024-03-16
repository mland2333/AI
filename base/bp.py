path = r'C:\Users\28929\Desktop\data.csv'
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
warnings.filterwarnings("ignore")

# 读取数据
df = pd.read_csv(path)
# 将数据划分为特征和标签
X = df.iloc[:, :-3]
y = df.iloc[:, -3:]
#print(y)

# 数据归一化
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=70)

train_score = 0
test_score = 0

N = 15
#可视化坐标
x_coor = [0 for index in range(N)]
test_coor1 = [0 for index in range(N)]
test_coor2 = [0 for index in range(N)]
test_coor3 = [0 for index in range(N)]
train_coor = [0 for index in range(N)]
#结果不稳定，取N次平均
#更改参数有：hidden_layer_sizes、solve、max_iter
for j in range (N):
    #x_coor[j] = 50*(j+1) #max_iter
    x_coor[j] = 2*(j+1) #hidden_layer_sizes
    train_score = 0
    test_score1 = 0
    test_score2 = 0
    test_score3 = 0
    i_iter = 5
    for i in range(i_iter):
        model1 = MLPClassifier(hidden_layer_sizes=(2+j*2, 8), activation='relu', 
                      solver='adam', max_iter=300)#定义模型
        model1.fit(X_train, y_train)#训练模型
        train_score+=model1.score(X_train, y_train)
        test_score1+=model1.score(X_test, y_test)#评分
        '''model2 = MLPClassifier(hidden_layer_sizes=(22, 10), activation='relu', 
                      solver='sgd', max_iter=50*(j+1))
        model2.fit(X_train, y_train)
        
        test_score2+=model2.score(X_test, y_test)
        model3 = MLPClassifier(hidden_layer_sizes=(22, 10), activation='relu', 
                      solver='lbfgs', max_iter=50*(j+1))
        model3.fit(X_train, y_train)
        
        test_score3+=model3.score(X_test, y_test)'''
    test_coor1[j] = test_score1/i_iter
    #test_coor2[j] = test_score2/i_iter
    #test_coor3[j] = test_score3/i_iter
    train_coor[j] = train_score/i_iter

plt.plot(x_coor,test_coor1,'s-',color = 'r',label="test_accuracy")
#plt.plot(x_coor,test_coor1,'s-',color = 'r',label="adam")#s-:方形
#plt.plot(x_coor,test_coor2,'o-',color = 'g',label="sgd")
#plt.plot(x_coor,test_coor3,'^-',color = 'b',label="lbfgs")
plt.plot(x_coor,train_coor,'o-',color = 'g',label="train_accuracy")
#plt.plot(x,k2,'o-',color = 'g',label="CNN-RLSTM")#o-:圆形
plt.xlabel("The number of neurons in the first hidden layer")#横坐标名字
plt.ylabel("accuracy")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()
