# -*- coding: utf-8 -*-
__author__ = 'wangwenjie'
__data__ = '2018/5/11 下午2:21'
__product__ = 'PyCharm'
__filename__ = 'random_forest_classifier'

"""随机森林"""



import matplotlib.pyplot as plt

from mglearn import plots
from mglearn import discrete_scatter

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

X_train, X_text, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)

fig, axes = plt.subplot(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title('tree {}'.format(i))
    plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], alpha=0.4)
axes[-1, -1].set_title("random froest")
discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)


