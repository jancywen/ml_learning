# -*- coding: utf-8 -*-
__author__ = 'wangwenjie'
__data__ = '2018/5/14 下午1:38'
__product__ = 'PyCharm'
__filename__ = 'grid_search_valid'


"""网格搜索 验证集 """
"""
    利用验证集选定最佳参数, 利用最佳参数设置重新构建一个模型,要同时在训练数据和验证数据上进行训练
    
"""


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()

X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=0)

best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_valid, y_valid)
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}

svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)
print("Best score on validation set: {:.2f}".format(best_score))
print('Best parameter: ', best_parameters)
print('Test set score with best parameters: {:.2f}'.format(test_score))