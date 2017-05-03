'''
performs simple classification models on the dataset as baseline
does not use lyrics
   
@author - Farzan Memarian
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymysql.cursors
import sys, os
from tqdm import tqdm
import time
sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )

from util import data_accessor_util
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# ----------------
#    models
# ----------------
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.gaussian_process.kernels import RBF

DEBUG = 1

# reading the data from sql table using the get_all_data method
X_train, y_train, train_le, X_test, y_test, test_le = data_accessor_util.get_all_data_sets()

# X = df.drop(['genre'], axis=1)
# y = df['genre']
# le = preprocessing.LabelEncoder()
# y = le.fit_transform(y)
X_train = X_train.drop('songID', axis=1)
X_test = X_test.drop('songID', axis=1)
h = .02  # step size in the mesh
# list of methods used, they correspond to the classifiers list
methods = ["Logistic Regression","Decision Tree", "Random Forest", 
         "Linear SVM", "RBF SVM", "Neural Net, MLP", 
         "Gaussian Process", "Gaussian Naive Bayes", "QDA","AdaBoost"]
# add these two methods as well
# Xgboost
# Gradient boosting

classifiers = [
    LogisticRegression(),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=200, max_features=10),
    MLPClassifier(alpha=1),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    AdaBoostClassifier()
    # KNeighborsClassifier(3)
    ]

# initialize the variable storing accuracy of methods
accuracy_resuts = []
# initialize the counter
i = 0 

# create an empty setore_accuracy.txt file 
file = open("store_accuracy.txt","w")
file.write("Storing the accuracy of different classification methods \n")
file.close()

# iterate over classifiers
for name, clf in tqdm(zip(methods, classifiers)):
    if DEBUG == 1: print "now performing: ", name
    i += 1
    start = time.time()
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predict)
    accuracy_resuts.append((name, accuracy))
    end = time.time()
    elasped_time = end - start
    print "execution time of %s was %s seconds" %(name, elasped_time)
    print "accuracy of method %s is %s" % (name, accuracy)
    entry = "accuracy of " + name + " = " + str(accuracy) + "\n"
    file = open("store_accuracy.txt","a")
    file.write(entry)
    file.close()
