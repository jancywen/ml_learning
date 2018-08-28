# -*- coding: utf-8 -*-

"""
*********************************************

    非线性分类

*********************************************
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC, SVC

from SupportVectorMachines.save_figure import save_fig

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'bs')
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'g^')
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r'$x_1$', fontsize=20)
    plt.ylabel(r'$x_2$', fontsize=20, rotation=0)


def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)




if __name__ == '__main__':
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

    # plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    # plt.show()

    """
    polynomial_svm_clf = Pipeline([
        ('poly_features', PolynomialFeatures(degree=3)),
        ('scaler', StandardScaler()),
        ('svm_clf', LinearSVC(C=10, loss='hinge', random_state=42))
    ])

    polynomial_svm_clf.fit(X, y)

    plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    # save_fig("moons_polynomial_svc_plot")
    plt.show()
    """


    poly_kernel_svm_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))
    ])
    poly_kernel_svm_clf.fit(X, y)

    poly100_kernel_svm_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm_clf', SVC(kernel='poly', degree=10, coef0=100, C=5))
    ])
    poly100_kernel_svm_clf.fit(X, y)

    plt.figure(figsize=(11, 4))

    plt.subplot(121)
    plot_predictions(poly_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.title(r'$d=3, r=1, C=5', fontsize=18)

    plt.subplot(122)
    plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.title(r'$d=10, r=10, C=5', fontsize=18)

    save_fig('moons_kernelized_polymomial_svc_plot')
    plt.show()