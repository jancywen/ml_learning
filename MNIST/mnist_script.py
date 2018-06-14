# -*- coding: utf-8 -*-
__author__ = 'wangwenjie'
__data__ = '2018/6/12 下午3:22'
__product__ = 'PyCharm'
__filename__ = 'mnist_script'


# from sklearn.datasets import fetch_mldata
#
# mnist = fetch_mldata('MNIST original')
# print(mnist)

import os
import struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


X_train, y_train = load_mnist("data")
X_test, y_test = load_mnist("data", kind="t10k")

X = np.vstack((X_train, X_test))
y = np.concatenate((y_train, y_test))


some_digit = X[3500]

some_digit_image = some_digit.reshape(28, 28)


# for i in range(len(X_train)):
#     if y[i] == 5:
#         print(i)


# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
# plt.axis('off')
# plt.show()

""" Binary Classification """

"""
y_train_5 = (y_train == 5)
y_test_5 = y_test == 5


# 随机梯度下降
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, tol=None, random_state=42)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([some_digit]))

# 交叉验证
from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy'))

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)


# 混淆矩阵
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train_5, y_train_pred))

# 精准度 召回率 f1
from sklearn.metrics import precision_score, recall_score, f1_score
print('精准度:', precision_score(y_train_5, y_train_pred))
print('召回率:', recall_score(y_train_5, y_train_pred))
print('f1:', f1_score(y_train_5, y_train_pred))


y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')


# PR
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

#
def plot_precision_recall_vc_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.legend()
    plt.ylim([0, 1])


# plot_precision_recall_vc_threshold(precisions, recalls, thresholds)
# plt.show()

def plot_precision_vc_recall(precisions, recalls):
    plt.plot(recalls[:-1], precisions[:-1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1])

# plot_precision_vc_recall(precisions, recalls)
# plt.show()


# ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, _ = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# plot_roc_curve(fpr, tpr)
# plt.show()

print('SGD auc:',roc_auc_score( y_train_5, y_scores))



from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_froest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method='predict_proba')
y_scores_forest = y_probas_froest[:, 1]
fpr_forest, tpr_forest, _ = roc_curve(y_train_5, y_scores_forest)

# plt.plot(fpr, tpr, 'b:', label='SGD')
# plot_roc_curve(fpr_forest, tpr_forest, 'Random Forest')
# plt.legend(loc='lower right')
# plt.show()

print('forest auc:', roc_auc_score(y_train_5, y_scores_forest))

"""


""" Multiclass Classification """


"""

# 随机梯度下降
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, tol=None, random_state=42)
# sgd_clf.fit(X_train, y_train)
# print(sgd_clf.predict([some_digit]))
# some_digit_score = sgd_clf.decision_function([some_digit])
# print(some_digit_score)


# OVO
# from sklearn.multiclass import OneVsOneClassifier
# ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, tol=None, random_state=42))
# ovo_clf.fit(X_train, y_train)
# print(ovo_clf.predict([some_digit]))
# print(len(ovo_clf.estimators_))

# Random Forest
# from sklearn.ensemble import RandomForestClassifier
# forest_clf = RandomForestClassifier(random_state=42)
# forest_clf.fit(X_train, y_train)
# print(forest_clf.predict_proba([some_digit]))

from sklearn.model_selection import cross_val_score, cross_val_predict
# sgd_cross_val_score = cross_val_score(sgd_clf, X_train, y_train, scoring='accuracy')
# print(sgd_cross_val_score)

from sklearn.preprocessing import StandardScaler
X_train_scaled = StandardScaler().fit_transform(X_train.astype(np.float64))
# sgd_cross_val_score_scaled = cross_val_score(sgd_clf, X_train_scaled, y_train, scoring='accuracy')
# print(sgd_cross_val_score_scaled)


y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)

from sklearn.metrics import confusion_matrix
conf_max = confusion_matrix(y_train, y_train_pred)
print(conf_max)

# plt.matshow(conf_max, cmap=plt.cm.gray)
# plt.show()


row_sums = conf_max.sum(axis=1, keepdims=True)
norm_conf_mx = conf_max / row_sums
np.fill_diagonal(norm_conf_mx, 0)
print(norm_conf_mx)

# plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# plt.show()


def plot_digits(imgs, **kwargs):
    plt.subplot()
    pass


cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
plt.figure(figsize=(8, 8))
# plt.subplot(221)
# plot_digits(X_aa[:25], images_per_row=5)
# plt.subplot(222)
# plot_digits(X_ab[:25], images_per_row=5)
# plt.subplot(223)
# plot_digits(X_ba[:25], images_per_row=5)
# plt.subplot(224)
# plot_digits(X_bb[:25], images_per_row=5)
# plt.show()
"""


""" Exercise """

from sklearn.neighbors import KNeighborsClassifier
# knn_clf = KNeighborsClassifier(n_neighbors=3)
# knn_clf.fit(X_train, y_train)
# k_score = knn_clf.score(X_test, y_test)
#
# print(k_score)

from sklearn.model_selection import train_test_split


X_trainval, X_train_val, y_trainval, y_train_val = X_train[:50000], X_train[50000:], y_train[:50000], y_train[50000:]

best_score = 0.0
for n in [1, 2, 3, 4, 5]:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_trainval, y_trainval)
    score = knn.score(X_train_val, y_train_val)
    if score > best_score:
        best_score = score
        best_para = {'n_neighbors': n}


knn = KNeighborsClassifier(**best_para)
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)
print(score)