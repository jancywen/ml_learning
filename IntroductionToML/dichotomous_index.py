# -*- coding: utf-8 -*-
__author__ = 'wangwenjie'
__data__ = '2018/6/6 上午10:29'
__product__ = 'PyCharm'
__filename__ = 'dichotomous_index'


""" 二分类指标 """

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
y = digits.target == 9

X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)

from sklearn.dummy import DummyClassifier

# 最常见预测
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)
print('Unique predicted labels: {}'.format(np.unique(pred_most_frequent)))
print('Frequent test score: {:.2f}'.format(dummy_majority.score(X_test, y_test)))

# 决策树
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
print('Tree test score: {:.2f}'.format(tree.score(X_test, y_test)))


# 随机预测
dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
print('Dummy test score: {:.2f}'.format(dummy.score(X_test, y_test)))

# 逻辑回归
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print('Logreg test score: {:.2f}'.format(logreg.score(X_test, y_test)))

print('\n********************我是分割线********************\n')

# 混淆矩阵
from sklearn.metrics import confusion_matrix

# confusion = confusion_matrix(y_test, pred_logreg)
# print('Confusion matrix: \n{}'.format(confusion))

print('Most frequent class:\n{}'.format(confusion_matrix(y_test, pred_most_frequent)))
print('Dummy model:\n{}'.format(confusion_matrix(y_test, pred_dummy)))
print('Decision tree: \n{}'.format(confusion_matrix(y_test, pred_tree)))
print('Logistic Regression: \n{}'.format(confusion_matrix(y_test, pred_logreg)))
print('\n********************我是分割线********************\n')


# f1 score
from sklearn.metrics import f1_score
# UndefinedMetricWarning
# print('Most frequent f1 score :{:.2f}'.format(f1_score(y_test, pred_most_frequent)))
print('Dummy model f1 score: {:.2f}'.format(f1_score(y_test, pred_dummy)))
print('Decision tree f1 score: {:.2f}'.format(f1_score(y_test, pred_tree)))
print('Logistic Regression f1 score: {:.2f}'.format(f1_score(y_test, pred_logreg)))
print('\n********************我是分割线********************\n')


# classification report
from sklearn.metrics import classification_report

# UndefinedMetricWarning
# print('Most frequent: \n{}'.format(classification_report(y_test, pred_most_frequent, target_names=["not nine", "nine"])))
print('Dummy model:\n{}'.format(classification_report(y_test, pred_dummy, target_names=["not nine", "nine"])))
print('Decision tree: \n{}'.format(classification_report(y_test, pred_tree, target_names=["not nine", "nine"])))
print('Logistic Regression: \n{}'.format(classification_report(y_test, pred_logreg, target_names=["not nine", "nine"])))
print('\n********************我是分割线********************\n')


# 考虑不确定性
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=400, centers=2, cluster_std=7.0, center_box=(-10.0, 10.0), shuffle=False, random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=0.5).fit(X_train, y_train)
print(classification_report(y_test, svc.predict(X_test)))

# 减小阈值 增大回收率
y_pred_lower_threshold = svc.decision_function(X_test) > -0.8
print(classification_report(y_test, y_pred_lower_threshold))

# 准确率-召回率曲线

# ROC曲线和AUC
