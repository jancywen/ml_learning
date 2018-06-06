# -*- coding: utf-8 -*-
__author__ = 'wangwenjie'
__data__ = '2018/6/5 上午10:54'
__product__ = 'PyCharm'
__filename__ = 'univariate_statistics'


''' 单变量统计 '''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

# 获得确定性的随机数
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))

# 向数据中添加噪声特征
#
X_w_noise = np.hstack([cancer.data, noise])

X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=.5)

# 使用 SelectPercentile来选择50% 的特征
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)

# 对训练集进行变换
X_train_selected = select.transform(X_train)

print('X_train.shape: {}'.format(X_train.shape))
print('X_train_selected.shape:{}'.format(X_train_selected.shape))

mask = select.get_support()
print(mask)

# plt.matshow(mask.reshape(1, -1), cmap='gray_r')
# plt.xlabel('Sample index')
# plt.show()

from sklearn.linear_model import LogisticRegression

X_test_selected = select.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train, y_train)
print('Score with all features: {:.3f}'.format(lr.score(X_test, y_test)))
lr.fit(X_train_selected, y_train)
print('Score with only selected feature: {:.3f}'.format(lr.score(X_test_selected, y_test)))


