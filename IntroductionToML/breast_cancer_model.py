# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

path = "datas/breast-cancer-wisconsin.data"
names = ['id','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',
         'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei',
        'Bland Chromatin','Normal Nucleoli','Mitoses','Class']

df = pd.read_csv(path, header=None, names=names)
datas = df.replace('?', np.nan).dropna(how='any')
# print(datas.head())
#
# print(df.info())

print(names[1:10])
print(names[10])

X = datas[names[1:10]]
y = datas[names[10]]
print(X.head())
print(y.head())