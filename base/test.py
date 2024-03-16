import matplotlib.pyplot as plt
from matplotlib.pyplot import axis
import numpy as np
#plt.axis([-10, 10, -1, 1])
axis('on')
def sigmoid(x):
    return 1. / (1. + np.exp(-x))
 
def plot_sigmoid():
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    print(x,y)
    plt.show()

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
 
def plot_tanh():
    x=np.arange(-10,10,0.1)
    y=tanh(x)
    plt.plot(x,y)
    plt.show()
def relu(x):
    return np.maximum(0,x)
 
def plot_relu():
    x=np.arange(-10,10,0.1)
    y=relu(x)
    plt.plot(x,y)
    plt.show()
plot_relu()
