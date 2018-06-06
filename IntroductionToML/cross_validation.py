# -*- coding: utf-8 -*-
__author__ = 'wangwenjie'
__data__ = '2018/5/14 上午10:42'
__product__ = 'PyCharm'
__filename__ = 'cross_validation'


"""交叉验证"""
"""
    k折交叉验证  参数: cv 折数

"""

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# from mglearn.plot_cross_validation import plot_stratified_cross_validation
#
# plot_stratified_cross_validation()


iris = load_iris()

logreg = LogisticRegression()

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print(scores)

"""交叉验证分离器"""
from sklearn.model_selection import KFold
""" 
    n_splits: 折数
    shuffle: 是否将数据打乱分层
    random_state: 获得可重复的打乱结果
"""
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
print('Cross-validation with KFold scores: \n{}'.format(cross_val_score(logreg, iris.data, iris.target, cv=kfold)))


"""留一法交叉验证"""

"""
    用于小数据
"""
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print('留一法交叉验证 mean accuracy: {:.2f}'.format(scores.mean()))


"""打乱交叉验证"""
"""
    test_size: 测试集取点
    train_size: 训练集
    n_splits: 
"""
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print('ShuffleSplit scores: \n{}'.format(scores))


"""分组交叉验证"""
"""
"""
from sklearn.model_selection import GroupKFold
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=12, random_state=0)
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
print('GroupKFold scores: {}'.format(scores))