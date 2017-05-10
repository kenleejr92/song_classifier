from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import sys, os
sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import load_data
from sklearn.metrics import accuracy_score


X_train, X_test, y_train, y_test = load_data.load_all()

print "done loading"

clf = RandomForestClassifier()

clf.fit(X_train,y_train)
preds = clf.predict(X_test)

print accuracy_score(y_test, preds)