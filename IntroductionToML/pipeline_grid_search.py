# -*- coding: utf-8 -*-
__author__ = 'wangwenjie'
__data__ = '2018/5/15 上午9:53'
__product__ = 'PyCharm'
__filename__ = 'pipeline_grid_search'



"""在网格搜索中使用管道"""



from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

"""
    pipeline 本身具有 fit predict score 方法
    构建一个由步骤列表组成的管道对象: 每个步骤都是一个元组, 包含一个名称(字符串) 和一个估计器实例
"""
pipe = Pipeline([("scale", MinMaxScaler()), ('svm', SVC())])

"""
    管道定义参数网格语法: 参数指定步骤名称,加 "__"(双下划线) , 加参数名称
"""
param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100], 'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
"""
    任何交叉验证都应该位于处理过程的"最外层循环" 防止信息泄露
"""
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

print('Best cross-validation accuracy: {:.2f}'.format(grid.best_score_))
print('Best parameter: {}'.format(grid.best_params_))
print('Test set score: {:.2f}'.format(grid.score(X_test, y_test)))



"""
    网格搜索使用哪个模型
"""

from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])

param_grid = [{'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
               'classifier__gamma':[0.001, 0.01, 0.1, 1, 10, 100],
               'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
              {'classifier': [RandomForestClassifier()], 'preprocessing': [None],
               'classifier__max_features':[1, 2, 3]}]
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print('Best parameter: {}'.format(grid.best_params_))
print('Best cross-validation accuracy: {:.2f}'.format(grid.best_score_))
print('Test-set score : {:.2f}'.format(grid.score(X_test, y_test)))
