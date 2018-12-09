from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import classification_report,confusion_matrix


X_train = np.genfromtxt("X_train.txt",delimiter=None)
Y_train = np.genfromtxt("Y_train.txt",delimiter=None)

X_test = np.genfromtxt("X_test.txt",delimiter=None)

X_train = np.delete(X_train,[0],1)
X_test = np.delete(X_test,[0],1)
clf = ExtraTreesRegressor(n_estimators=700, n_jobs=-1, max_features=10, min_samples_leaf=4, max_depth=30) # was 3

clf.fit(X_train, Y_train)

Ypred = clf.predict(X_test)

np.savetxt('Y_submit.txt',np.vstack( (np.arange(len(Ypred)) , Ypred) ).T, '%d, %.2f',header='ID,Prob1',comments='',delimiter=',');