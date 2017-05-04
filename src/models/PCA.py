
"""
PCA model

- Performs PCA for feature reduction

@author - Farzan Memarian
"""

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
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
matplotlib.use('Agg')


DEBUG = 1

# reading the data from sql table using the get_all_data method

X_train, y_train, train_le, X_test, y_test, test_le = data_accessor_util.get_all_data_sets()
classes = list(test_le.classes_)
# print test_le.inverse_transform([0, 1, 2, 3, 4, 5, 6])

# Converting data to numpy
(X_train, y_train, X_test, y_test) = data_accessor_util.convert_data_sets_to_numpy(X_train, y_train, X_test, y_test)


pca = PCA(n_components=175)
pca.fit(X_train)
var = pca.explained_variance_ratio_
fig = plt.figure(1)
plt.plot(var)
plt.title('individual scree plot')
plt.xlabel('principal components')
plt.ylabel('proportion of variance explained')
plt.savefig("proportionOfVariance.jpg")

commulated_var = np.zeros(175)
for i in range(50):
    commulated_var[i] = np.sum(var[:i])

fig2 = plt.figure(2)
plt.plot(commulated_var)
plt.title('commulative scree plot')
plt.xlabel('principal components')
plt.ylabel('commulative proportion of variance explained')    
plt.savefig("proportionOfVarianceCom.jpg")
