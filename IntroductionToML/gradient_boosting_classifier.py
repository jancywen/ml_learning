# -*- coding: utf-8 -*-
__author__ = 'wangwenjie'
__data__ = '2018/5/11 下午2:59'
__product__ = 'PyCharm'
__filename__ = 'gradient_boosting_classifier'


"""梯度提升回归树"""


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1, n_estimators=100, learning_rate=0.1).fit(X_train, y_train)


print('Grbt train score:{:.3f}'.format(gbrt.score(X_train, y_train)))
print('grbt train score {:.3f}'.format(gbrt.score(X_test, y_test)))

def plot_feature_importance_cancer(model):
    n_feature = cancer.data.shape[1]
    plt.barh(range(n_feature), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_feature), cancer.feature_names)
    plt.show()


plot_feature_importance_cancer(gbrt)