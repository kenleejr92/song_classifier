from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import sys, os
sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util
from sklearn.metrics import accuracy_score


X_train, y_train, train_le, X_test, y_test, test_le = data_accessor_util.get_all_data_sets()

classes = list(test_le.classes_)
# print test_le.inverse_transform([0, 1, 2, 3, 4, 5, 6])

# Converting data to numpy
(X_train, y_train, X_test, y_test) = data_accessor_util.convert_data_sets_to_numpy(X_train, y_train, X_test, y_test)

np.save('X_train.npy',X_train)
np.save('X_test.npy',X_test)
np.save('y_train.npy',y_train)
np.save('y_test.npy',y_test)