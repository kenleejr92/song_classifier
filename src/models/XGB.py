"""
XGBClassifier model

- Performs XGBClassifier to classify the data

@author - Farzan Memarian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymysql.cursors
import sys, os
import time
from tqdm import tqdm
sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util
from util import run_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from util import PCA
from xgboost import XGBClassifier


# ----------------
#    models
# ----------------
method = "XGBClassifier"
possibles = globals().copy()
possibles.update(locals())
method_func = possibles.get(method)
file_name = method + ".txt"

# setting the model parameters
clf = method_func(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='multi:softmax', \
nthread=5, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, \
colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None)

# use a full grid search over following parameters
param_grid = {"max_depth": [3, 10],\
              "n_estimators": [100, 200]}

run_model.run_model(clf, param_grid, method, file_name)