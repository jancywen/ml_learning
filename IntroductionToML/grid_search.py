# -*- coding: utf-8 -*-
__author__ = 'wangwenjie'
__data__ = '2018/5/14 下午1:20'
__product__ = 'PyCharm'
__filename__ = 'grid_search'

"""网格搜索"""


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
print('Size of training set: {} ; size of test set: {}'.format(X_train.shape[0], X_test.shape[0]))

best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}

print('Best score : {:.2f}'.format(score))
print('Best parameters: {}'.format(best_parameters))