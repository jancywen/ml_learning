# -*- coding: utf-8 -*-
__author__ = 'wangwenjie'
__data__ = '2018/6/5 下午1:50'
__product__ = 'PyCharm'
__filename__ = 'expert_knowledge'


''' 专家知识 '''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def load_citibike():
    data_mine = pd.read_csv(os.path.join('datas', "JC-201608-citibike-tripdata.csv"))
    data_mine['one'] = 1
    data_mine['starttime'] = pd.to_datetime(data_mine['Start Time'])
    data_starttime = data_mine.set_index("starttime")
    data_resampled = data_starttime.resample("3h").sum().fillna(0)
    return data_resampled.one


citibike = load_citibike()
# print(type(citibike))
# print(citibike.head())

# plt.figure(figsize=(10, 4.5))
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(), freq='D')
# # print(xticks)
# plt.xticks(xticks, xticks.strftime('%a, %m-%d'), rotation=90, ha='left')
# plt.plot(citibike, linewidth=1)
# plt.xlabel('Date')
# plt.ylabel('Rentals')
# plt.show()



y = np.array(citibike.values)
X = np.array(list(citibike.index.strftime('%s'))).astype(np.int).reshape(-1, 1)
# print(X)
# print(np.arange(0, len(X), 8))
# print(np.arange(10))

# print(type(xticks))
# print(xticks)
xticks = np.array(list(xticks.strftime('%a, %m-%d')))
# print(len(xticks))
# print(len(np.arange(0, len(X), 8)))
#
# print(len(y))
# print(type(y))
n_train = 184

# X_train, X_test = X[:n_train], X[n_train:]
# y_train, y_test = y[:n_train], y[n_train:]
#
#
# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators=100, random_state=0)
# regressor.fit(X_train, y_train)
# print('Test set R^2: {:.3f}'.format(regressor.score(X_test, y_test)))
# y_pred = regressor.predict(X_test)
# y_pred_train = regressor.predict(X_train)
#
# plt.xticks(np.arange(0, len(X), 8), xticks, rotation=90)
# plt.plot(np.arange(len(X)), y, '-')
# plt.plot(np.arange(n_train), y_train, '--', label='train')
# plt.plot(np.arange(n_train, len(y_test) + n_train), y_test, '-', label='test')
# plt.plot(np.arange(n_train), y_pred_train, '--', label='prediction train')
# plt.plot(np.arange(n_train, len(y_test) + n_train), y_pred, '--', label='prediction test')
# plt.legend(loc=(1.01, 0))
# plt.show()



def eval_on_features(features, target, regressor):
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)
    print('Test set R^2: {:.3f}'.format(regressor.score(X_test, y_test)))
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    plt.figure(figsize=(10, 3))
    plt.xticks(np.arange(0, len(X), 8), xticks, rotation=90)
    # plt.plot(np.arange(len(features)), y, '-')
    plt.plot(np.arange(n_train), y_train, label='train')
    plt.plot(np.arange(n_train, len(y_test) + n_train), y_test, '-', label='test')
    plt.plot(np.arange(n_train), y_pred_train, '--', label='prediction train')
    plt.plot(np.arange(n_train, len(y_test) + n_train), y_pred, '--', label='prediction test')
    plt.legend(loc=(1.01, 0))
    plt.xlabel('Date')
    plt.ylabel('Rentals')
    plt.title('Test set R^2: {:.3f}'.format(regressor.score(X_test, y_test)))
    plt.show()

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)

# eval_on_features(X, y, regressor)

X_hour = np.array(list(citibike.index.hour)).astype(np.int).reshape(-1, 1)
# eval_on_features(X_hour, y, regressor)

X_week = np.array(list(citibike.index.strftime('%s'))).astype(np.int).reshape(-1, 1)
X_hour_week = np.hstack([X_week, X_hour])
# eval_on_features(X_hour_week, y, regressor)

from sklearn.linear_model import LinearRegression
# eval_on_features(X_hour_week, y, LinearRegression())

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()

from sklearn.linear_model import Ridge
# eval_on_features(X_hour_week_onehot, y, Ridge())


from sklearn.preprocessing import PolynomialFeatures
poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)
lr = Ridge()
eval_on_features(X_hour_week_onehot_poly, y, lr)

# hour = ["%02d:00" % i for i in range(0, 24, 3)]
# day = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
# features = day + hour
#
# print(features)
#
# features_poly = poly_transformer.get_feature_names()
# print(features_poly)
# features_nonzero = np.array(features_poly)[lr.coef_ != 0]
# coef_nonZero = lr.coef_[lr.coef_ != 0]
# plt.figure(figsize=(15, 2))
# plt.xticks(np.arange(len(coef_nonZero)), coef_nonZero, rotation=90)
# plt.plot(coef_nonZero, 'o')
# plt.xlabel('feature name')
# plt.ylabel('Feature magnitude')
# plt.show()
