# -*- coding: utf-8 -*-

"""
@author:
@brief: 用于将多个标准takonize
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class TagsNum(BaseEstimator, TransformerMixin):
#     def __init__(self):

    def fit(self, X, y=None):
        return self

    def transform(self, tags):
        tag_num = []
        for tag in tags:
            tag_num.append(len(tag.split(',')))
        return np.mat(tag_num).T


