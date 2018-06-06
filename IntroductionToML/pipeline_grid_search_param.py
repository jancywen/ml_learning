# -*- coding: utf-8 -*-
__author__ = 'wangwenjie'
__data__ = '2018/5/15 下午1:11'
__product__ = 'PyCharm'
__filename__ = 'pipeline_grid_search_param'


"""
    
"""

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt


boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())

param_grid = {'polynomialfeatures__degree': [1, 2, 3], 'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)

grid.fit(X_train, y_train)

print('Best parameter: \n{}'.format(grid.best_params_))
print('Test-set score: \n{:.2f}'.format(grid.score(X_test, y_test)))


plt.matshow(grid.cv_results_['mean_test_score'].reshape(3, -1), vmin=0, cmap="viridis")
plt.xlabel('ridge__alpha')
plt.ylabel('polynomicalfeatures__degree')
plt.xticks(range(len(param_grid['ridge__alpha'])), param_grid['ridge__alpha'])
plt.yticks(range(len(param_grid['polynomialfeatures__degree'])), param_grid['polynomialfeatures__degree'])
plt.colorbar()
plt.show()
