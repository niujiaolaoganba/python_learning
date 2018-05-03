from hyperopt import fmin, tpe, hp, STATUS_OK, rand, Trials
import networkx
import time
import pickle
from hyperopt.mongoexp import MongoTrials
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import roc_auc_score

import networkx

train = pd.read_csv('/Users/finup/Documents/Finup/Finup_lgb/data/qz_andr.csv')

X = train['tags']
y = train['is_reg']

tag_vector = CountVectorizer(tokenizer=lambda x: x.split(','))
features = tag_vector.fit(X).transform(X).toarray()
X = pd.DataFrame(features, columns=tag_vector.get_feature_names())

X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size = 0.2, random_state = 2018)

print(X.shape)

def obj(param):
    learner = LogisticRegression(**param)
    learner.fit(X_train, y_train)
    if hasattr(learner, 'predict_proba'):
        y_pred = learner.predict_proba(X_valid)[:,1]
    else :
        y_pred = learner.predict(X_valid)
    loss = - roc_auc_score(y_valid, y_pred)
    ret = {
            "loss": loss,
            "status": STATUS_OK,
        }
    print(param, ret)
    return ret