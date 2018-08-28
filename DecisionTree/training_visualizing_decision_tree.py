# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from DecisionTree.tool import *

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data[:, 2:]
    y = iris.target
    tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree_clf.fit(X, y)

    export_graphviz(
        tree_clf,
        out_file=image_path('iris_tree.dot'),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )