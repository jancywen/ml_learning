# -*- coding: utf-8 -*-
__author__ = 'wangwenjie'
__data__ = '2018/5/15 上午10:42'
__product__ = 'PyCharm'
__filename__ = 'pipeline_common_port'


"""
    通用的管道接口
    
    1,对管道中的估计器的唯一要求:
        除了最后一步之外的所有步骤都需要具有transform方法, 这样他们可以生产新的数据表示,以供下一个步骤使用
    2,在调用Pipeline.fit的过程中,管道内部依次对每个步骤调用fit和transform,其输入是前一步骤的输出,最后一步仅调用fit
    3,make_pipeline: 创建管道并根据每个步骤所属的类为其自动命名
        步骤名称只是类名称的小写版本, 如果多个版本属于同一个类,则会附加一个数字
    4,Pipeline.steps 所有步骤元组列表
    5,name_steps 步骤字典
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())
# print('Pipeline steps: \n{}'.format(pipe.steps))
# print('Pipeline name_steps: \n{}'.format(pipe.named_steps))


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)


pipe = make_pipeline(StandardScaler(), LogisticRegression())

param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

# best_estimator_ 是一个管道
print('Best estimator: \n{}'.format(grid.best_estimator_))
print('Logistic regression step: \n{}'.format(grid.best_estimator_.named_steps['logisticregression']))
print('Logistic regression coefficient: \n{}'.format(grid.best_estimator_.named_steps['logisticregression'].coef_))

