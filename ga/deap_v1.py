# !/bin/python
# -*- coding: utf-8 -*-
"""
# 利用GA进行特征选择
"""

from deap import algorithms
from deap import creator
from deap import tools
from deap import base
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def eva_max(individual, x_train, y_train, x_test, y_test):
#    """
#    评价函数
#    :return: ouput_data,
#    """
    col_idx = [i for i in range(len(individual)) if individual[i] == 1]
    x_train = x_train[:, col_idx]
    x_test = x_test[:, col_idx]
    clf = LogisticRegression(C=0.0009, penalty='l2', max_iter=200, random_state=123)
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)[:,1]
    auc = roc_auc_score(np.array(y_test,dtype=int),y_pred)
    return auc,

def FeatureSelectGA(feature_num, x_train, y_train, x_test, y_test):
    """
    利用GA进行特征选择
    :return:
    """
    # 初始化
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # 初始种群生成
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=feature_num)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 进化器生成
    toolbox.register("evaluate", eva_max, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 进化过程演示
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("max", np.max)

    # 初始化种群并进行优化
    population = toolbox.population(n=200)
    best_model = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.1, ngen=50, stats=stats, halloffame=hof)
    return best_model[0]

if __name__ == '__main__':
    FeatureSelectGA(400, 1, 1, 1, 1)
