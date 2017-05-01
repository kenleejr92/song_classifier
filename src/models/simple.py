from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import sys, os
sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util
from sklearn.metrics import accuracy_score

df = data_accessor_util.get_all_data()

X = df.drop(['genre'], axis = 1)

y = df['genre']    # DataFrame storing the outputs
le = preprocessing.LabelEncoder()
y = pd.DataFrame(le.fit_transform(y))

print X.head()

msk = np.random.rand(len(df)) < 0.8
X_train = X[msk]
X_test = X[~msk]

y_train = y[msk]
y_test = y[~msk]

clf = LogisticRegression()

clf.fit(X_train,y_train)
preds = clf.predict(X_test)

print accuracy_score(y_test, preds)