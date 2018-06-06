# -*- coding: utf-8 -*-
__author__ = 'wangwenjie'
__data__ = '2018/5/31 上午9:58'
__product__ = 'PyCharm'
__filename__ = 'Housing'

import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit


DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = 'datasets/housing'
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + '/housing.tgz'


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):

    '''
    下载数据
    :param housing_url:
    :param housing_path:
    :return:
    '''

    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    '''
    读取本地数据
    :param housing_path:
    :return:
    '''
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    """
    数据分割
    :param data:
    :param test_ratio:
    :return:
    """
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[: test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


# fetch_housing_data()

housing = load_housing_data()
# print(housing.head())
# print(housing.info())
"""
# head() take a look at the top five rows
print(housing.head())
# info(): get a quick description of the data, in particular the total number of rows,
#  and each attribute's type and number of non-null values
print(housing.info())
# value_counts():  find out what categories exist and how many districts belong to each category
print(housing['ocean_proximity'].value_counts())
# describe(): summary of the numerical attributes
print(housing.describe())

housing.hist(bins=50, figsize=(20, 15))
plt.show()
"""

"""
train_set, test_set = split_train_test(housing, 0.2)

housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
"""

#
# 预处理 创建 income_cat 属性
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# 随机取样
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# 分层采样
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    # print("train_index:", train_index, "\n test_index:", test_index)
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


"""
#Sampling bias comparison of stratified versus purely random sampling
overAll = housing["income_cat"].value_counts(sort=False) / len(housing)
overAll.name = 'over_all'
strata = strat_test_set["income_cat"].value_counts(sort=False) / len(strat_test_set)
strata.name = 'strata'
stra_error = (strata - overAll) * 100 / overAll
stra_error.name = "stra % err"
random = test_set["income_cat"].value_counts(sort=False) / len(test_set)
random.name = 'random'
rand_error = (random - overAll) * 100 / overAll
rand_error.name = 'rand % err'
print(pd.concat([overAll, strata, random, stra_error, rand_error], axis=1))
"""

# remove the income_cat attribute
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


housing = strat_train_set.copy()

# 散点
# housing.plot(kind='scatter', x="longitude", y="latitude", alpha=0.1)
# plt.show()

"""
# s:圆圈半径, c:颜色, cmap: 颜色图
housing.plot(kind="scatter", x='longitude', y='latitude', alpha=0.4,
             s=housing['population']/100, label='population', c='median_house_value',
             cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
plt.show()
"""

"""
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()
"""

# Median income versus median house value
# housing.plot(kind='scatter', x="median_income", y="median_house_value", alpha=0.4, color='b')
# plt.show()


# attribute combination
# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"] = housing["population"]/housing["households"]
# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# print(housing)
# print(housing_labels)
print(housing.info())


""" Data clear """

# get rid of corresponding districts
housing.dropna(subset=["total_bedroms"])

# get rid of the whole attribute
housing.drop("total_bedrooms", axis=1)

# set the values to some value(zero, the mean, the median, etx.).
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
# housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# labelEncoder handling text and categorical attribute
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(encoder.classes_)