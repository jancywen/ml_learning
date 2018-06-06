# -*- coding: utf-8 -*-
__author__ = 'wangwenjie'
__data__ = '2018/5/11 上午11:15'
__product__ = 'PyCharm'
__filename__ = 'decision_tree_01'


"""
    决策树
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.tree import export_graphviz


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# 决策树
tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)

print('Tree Train score : %.3f' % tree.score(X_train, y_train))
print('Tree Test score: {:.3f}'.format(tree.score(X_test, y_test)))

# 逻辑回归
logistics = LogisticRegression().fit(X_train, y_train)

print('Regression train score: %.3f' % logistics.score(X_train, y_train))
print('Regression test score: %.3f' % logistics.score(X_test, y_test))


# export_graphviz(tree, out_file='tree.dot', class_names=["malignant", "benign"], feature_names=cancer.feature_names, impurity=False, filled=True)



# 随机森林

forest = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
print('RandomForestClassifier train score: %.3f' % forest.score(X_train, y_train))
print('RandomForestClassifier test score: %.3f' % forest.score(X_test, y_test))


# 梯度提升回归树
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1, n_estimators=100, learning_rate=0.01).fit(X_train, y_train)
print('Grbt train score:{:.3f}'.format(gbrt.score(X_train, y_train)))
print('grbt train score {:.3f}'.format(gbrt.score(X_test, y_test)))


def plot_feature_importance_cancer(model):
    n_feature = cancer.data.shape[1]
    plt.barh(range(n_feature), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_feature), cancer.feature_names)
    plt.show()

plot_feature_importance_cancer(tree)

plot_feature_importance_cancer(forest)

plot_feature_importance_cancer(gbrt)