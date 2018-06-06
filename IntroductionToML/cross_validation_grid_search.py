# -*- coding: utf-8 -*-
__author__ = 'wangwenjie'
__data__ = '2018/5/14 下午2:18'
__product__ = 'PyCharm'
__filename__ = 'cross_validation_grid_search'

"""带交叉验证的网格搜索"""


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

import numpy as np

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)


best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in[0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        scores = cross_val_score(svm, X_train, y_train, cv=5)
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}

svm = SVC(**best_parameters)
svm.fit(X_train, y_train)
test_score = svm.score(X_test, y_test)
print("Best score on validation set: {:.2f}".format(best_score))
print('Best parameter: ', best_parameters)
print('Test set score with best parameters: {:.2f}'.format(test_score))



from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print('Test score: {:.2f}'.format(grid_search.score(X_test, y_test)))
print('Best parameter: {}'.format(grid_search.best_params_))
print('Best cross-validation score: {:.2f}'.format(grid_search.best_score_))

results = pd.DataFrame(grid_search.cv_results_)
scores = np.array(results.mean_test_score).reshape(6, 6)

values = scores
xlabel='gamma'
xticklabels=param_grid['gamma']
ylabel='C'
yticklabels=param_grid['C']
cmap="viridis"
fmt="%0.2f"
vmin=None
vmax=None

ax = plt.gca()
# plot the mean cross-validation scores
img = ax.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax)
img.update_scalarmappable()
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_xticks(np.arange(len(xticklabels)) + .5)
ax.set_yticks(np.arange(len(yticklabels)) + .5)
ax.set_xticklabels(xticklabels)
ax.set_yticklabels(yticklabels)
ax.set_aspect(1)


for p, color, value in zip(img.get_paths(), img.get_facecolors(), img.get_array()):
    x, y = p.vertices[:-2, :].mean(0)
    if np.mean(color[:3]) > 0.5:
        c = 'k'
    else:
        c = 'w'
    ax.text(x, y, fmt % value, color=c, ha="center", va="center")

plt.show()
# def heatmap(values, xlabel, ylabel, xticklabels, yticklabels, cmap=None,
#             vmin=None, vmax=None, ax=None, fmt="%0.2f"):
#     if ax is None:
#
#     return img