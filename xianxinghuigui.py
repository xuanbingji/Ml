#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-10 17:46:07
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

data = pd.read_csv('D:/data/Advertising.csv', usecols=[1, 2, 3, 4])
# sns.pairplot(data,x_vars =['TV','radio','newspaper'],y_vars='sales',size
# =7,aspect=0.8,kind='reg')

# plt.show()
feature_cols = ['TV', 'radio']
X = data[feature_cols]
y = data['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg = LinearRegression()
model = linreg.fit(X_train, y_train)
print(model)
print(linreg.intercept_)
print(linreg.coef_)
y_pred = linreg.predict(X_test)
sum_mean = 0
for i in range(len(y_pred)):
    sum_mean += (y_pred[i] - y_test.values[i])**2
sum_erro = np.sqrt(sum_mean / 50)
print('RMSE by hand:', sum_erro)
plt.figure()
plt.plot(range(len(y_pred)), y_pred, 'b')
plt.plot(range(len(y_pred)), y_test, 'r')
plt.xlabel('the number of sales')
plt.ylabel('value of sales')
plt.legend()
plt.show()
