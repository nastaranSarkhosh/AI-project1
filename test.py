import math
import time

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from t import plotChart
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D


def cost_function(X, y, w):
    m = y.size
    error = np.dot(X, w) - y
    cost = 1 / (2 * m) * np.dot(error.T, error)
    return cost, error


def gradient_descent(X, y, w, a, loop):
    c_array = np.zeros(loop)
    m = y.size
    for i in range(loop):
        cost, error = cost_function(X, y, w)
        w = w - (a * (1 / m) * np.dot(X.T, error))
        c_array[i] = cost
    return w, c_array


df = pd.read_csv('./Flight_Price_Dataset_Q2.csv')
departureTimeMap = {
    'Early_Morning': 0,
    'Morning': 1,
    'Afternoon': 2,
    'Evening': 3,
    'Night': 4,
    'Late_Night': 5

}
stopsMap = {
    'zero': 0,
    'one': 1,
    'two_or_more': 2,
}
arrivalTimeMap = {
    'Early_Morning': 0,
    'Morning': 1,
    'Afternoon': 2,
    'Evening': 3,
    'Night': 4,
    'Late_Night': 5
}
classMap = {
    'Economy': 0,
    'Business': 1
}
file = open("11-UIAI4021-PR1-Q2.txt", "w+")
df['class'] = df['class'].map(classMap)
df['arrival_time'] = df['arrival_time'].map(arrivalTimeMap)
df['stops'] = df['stops'].map(stopsMap)
df['departure_time'] = df['departure_time'].map(departureTimeMap)
X_train = df[['departure_time', 'stops', 'arrival_time', 'class', 'duration', 'days_left']]
startTime=time.time()
X_train, X_test, y_train, y_test = train_test_split(
    df[['departure_time', 'stops', 'arrival_time', 'class', 'duration', 'days_left']], df['price'], test_size=0.2, shuffle=True)
a = 0.01
loop = 1000
X_train = (X_train - X_train.mean()) / X_train.std()#normalize()
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
w = np.zeros(X_train.shape[1])
w, cost_num = gradient_descent(X_train, y_train, w, a, loop)
# print(w)
X_test = (X_test - X_test.mean()) / X_test.std()
X_test = np.c_[np.ones(X_test.shape[0]), X_test]
y_pred = np.dot(X_test, w)

file.write("PRICE: %d * departure_time +  %d * stops + %d * arrival_time + %d * class + %d * duration + %d * days_left + %d\n" % (w[1],w[2],w[3],w[4],w[5],w[6],w[0]))
file.write("Training Time: %f s\n" % (math.ceil(time.time() - startTime)))
file.write("MSE : %d\n"%(mean_squared_error(y_test, y_pred)))
file.write("RMSE : %d\n"%(math.sqrt(mean_squared_error(y_test, y_pred))))
file.write("MAE : %d\n"%(mean_absolute_error(y_test, y_pred)))
file.write("R2: : %f\n"%(r2_score(y_test, y_pred)))

loop1=np.arange(loop)
plt.plot(loop1, cost_num)
plt.xlabel('loop')
plt.ylabel('cost')
plt.show()
