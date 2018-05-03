
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

import optimizer

param_space_clf_skl_lr = {
    "C": hp.uniform("C", 1e-5, 10),
    "penalty": hp.choice("penalty", ['l1', 'l2']),
    "random_state": 42,
}

trials = MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp3')
# trials = Trials()
best = fmin(fn = optimizer.obj, space = param_space_clf_skl_lr, algo = tpe.suggest, max_evals = 10, trials = trials)
print(best)
