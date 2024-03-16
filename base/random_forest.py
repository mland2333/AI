path = r'C:\Users\28929\Desktop\data.csv'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
import warnings
from scipy.interpolate import griddata
import plotly.graph_objects as go
warnings.filterwarnings("ignore")

df = pd.read_csv(path)

X = df.iloc[:, :-3]
y = df.iloc[:,-3:]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=70)
N = 15
x_coor1 = [0 for index in range(N*N)]
y_coor1 = [0 for index in range(N*N)]
z_coor1 = [0 for index in range(N*N)]

n=0
m=0
s=0
max_score=0
for i in range(N):
    for j in range(N):
        x_coor1[i*N+j] = (i+1)*5
        y_coor1[i*N+j] = 5+j
        model = RandomForestClassifier(n_estimators=(i+1)*5, max_depth=5+j, random_state=46)
        model.fit(X_train, y_train)
        z_coor1[i*N+j] = model.score(X_test, y_test)
        if model.score(X_test, y_test) > max_score:
            n,m,s=(i+1)*5,5+j,model.score(X_test, y_test)
print(n,m,s)
'''grid_x, grid_y = np.mgrid[min(x_coor1):max(x_coor1):100j, min(y_coor1):max(y_coor1):100j]  #生成（x，y）格点，其中100j表示从最小值到最大值之间插入100个点
Z = griddata(np.array([x_coor1,y_coor1]).T, z_coor1, (grid_x, grid_y), method='linear') #这里用的是线性插值，此外还可用其他插值方法，搜scipy.interpolate
fig=go.Figure(data=go.Heatmap(x=grid_x[:,1], y=grid_y[1,:], z=Z.T)) 
fig.show()'''
'''x_coor1=np.array(x_coor1)
y_coor1=np.array(y_coor1)
z_coor1=np.array(z_coor1)'''
#三维曲面图，根据离散的三维坐标生成连续图形
xi=np.linspace(min(x_coor1),max(x_coor1))
yi=np.linspace(min(y_coor1),max(y_coor1))
xi,yi=np.meshgrid(xi,yi)
zi=griddata(np.array([x_coor1,y_coor1]).T,z_coor1,(xi,yi),method='cubic')
fig=plt.figure()
ax = plt.axes(projection='3d')
surf=ax.plot_surface(xi,yi,zi,cmap='BuPu',linewidth=0,antialiased=False)
fig.colorbar(surf)
ax.set_xlabel('n_estimators')
ax.set_ylabel('max_depth')
ax.set_zlabel('accuracy')
plt.show()
