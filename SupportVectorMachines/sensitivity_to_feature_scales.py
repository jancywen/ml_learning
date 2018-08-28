# -*- coding: utf-8 -*-
"""
*********************************************

    SVC 对数据缩放敏感

*********************************************
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from SupportVectorMachines.save_figure import save_fig
from SupportVectorMachines.plot_svc_decision_boundary import plot_svc_decision_boundary

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


if __name__ == '__main__':
    Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)
    ys = np.array([0, 0, 1, 1])
    svm_clf = SVC(kernel='linear', C=100)
    svm_clf.fit(Xs, ys)

    plt.figure(figsize=(12, 3.2))
    plt.subplot(121)
    plt.plot(Xs[:, 0][ys == 1], Xs[:, 1][ys == 1], 'bo')
    plt.plot(Xs[:, 0][ys == 0], Xs[:, 1][ys == 0], 'ms')
    plot_svc_decision_boundary(svm_clf, 0, 6)
    plt.xlabel('$x_0$', fontsize=20)
    plt.ylabel('$x_1$', fontsize=20)
    plt.title('Unscaled', fontsize=16)
    plt.axis([0, 6, 0, 90])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xs)
    svm_clf.fit(X_scaled, ys)

    plt.subplot(122)
    plt.plot(X_scaled[:, 0][ys == 1], X_scaled[:, 1][ys == 1], 'bo')
    plt.plot(X_scaled[:, 0][ys == 0], X_scaled[:, 1][ys == 0], 'ms')
    plot_svc_decision_boundary(svm_clf, -2, 2)
    plt.xlabel('$x_0$', fontsize=20)
    plt.title('Scaled', fontsize=16)
    plt.axis([-2, 2, -2, 2])
    save_fig('sensitivity_to_feature_scale_plot')
    plt.show()




