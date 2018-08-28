# -*- coding: utf-8 -*-

"""
*********************************************

    大间距分类器

*********************************************
"""


import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn import datasets

from SupportVectorMachines.save_figure import save_fig
from SupportVectorMachines.plot_svc_decision_boundary import plot_svc_decision_boundary

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = iris["target"]

    setosa_or_versicolor = (y == 0) | (y == 1)
    X = X[setosa_or_versicolor]
    y = y[setosa_or_versicolor]

    svm_clf = SVC(kernel='linear', C=float("inf"))
    svm_clf.fit(X, y)
    # print(svm_clf)

    # Bad model
    x0 = np.linspace(0, 5.5, 200)
    pred_1 = 5 * x0 - 20
    pred_2 = x0 - 1.8
    pred_3 = 0.1 * x0 + 0.5

    plt.figure(figsize=(12, 2.7))

    plt.subplot(121)
    plt.plot(x0, pred_1, "g--")
    plt.plot(x0, pred_2, 'm-')
    plt.plot(x0, pred_3, 'r-')
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs', label='Iris-Versicolor')
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'yo', label='Iris-Setosa')
    plt.xlabel('Petal length', fontsize=14)
    plt.ylabel('Petal width', fontsize=14)
    plt.legend(loc='upper left', fontsize=14)
    plt.axis([0, 5.5, 0, 2])

    plt.subplot(122)
    plot_svc_decision_boundary(svm_clf, 0, 5.5)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'yo')
    plt.xlabel('Petal length', fontsize=14)
    plt.axis([0, 5.5, 0, 2])
    save_fig("large_margin_classification_plot")
    plt.show()
