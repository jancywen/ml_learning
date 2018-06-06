# -*- coding: utf-8 -*-
__author__ = 'wangwenjie'
__data__ = '2018/6/1 下午3:43'
__product__ = 'PyCharm'
__filename__ = 'principal_component_analysis'


from sklearn.datasets import fetch_lfw_people
import numpy as np


people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

print(people.images.shape)
print(image_shape)

# print(people)


counts = np.bincount(people.target)
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='    ')
    if (i + 1) % 3 == 0:
        print()

mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][0:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]
