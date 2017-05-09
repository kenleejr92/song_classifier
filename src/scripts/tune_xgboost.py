import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymysql.cursors
import sys, os
from   tqdm import tqdm
import time
sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize
from util import load_data


X_train, X_test, y_train, y_test, classes = load_data.load_all()


print "done loading"

classes = list(test_le.classes_)

(X_train, y_train, X_test, y_test) = data_accessor_util.convert_data_sets_to_numpy(X_train, y_train, X_test, y_test)

y_train = label_binarize(y_train, classes = classes)

print X_train.shape
print y_train.shape

# param_test1 = {
#  'max_depth':range(3,10,2),
#  'min_child_weight':range(1,6,2)
# }
# gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
#  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
#  param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

# print "starting grid search..."
# gsearch1.fit(X_train,y_train)
# print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_