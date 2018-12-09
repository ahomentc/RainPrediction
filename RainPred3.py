from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report,confusion_matrix

scaler = StandardScaler()

X_train = np.genfromtxt("X_train.txt",delimiter=None)
# X_test = np.genfromtxt("X_test.txt",delimiter=None)
Y_train = np.genfromtxt("Y_train.txt",delimiter=None)

X_train, X_test, Y_train, Y_test  = train_test_split(
		X_train,
		Y_train,
        train_size=0.80, 
        random_state=1234)


# trees
# clf = DecisionTreeClassifier(max_depth=6)

# clf = MLPRegressor(hidden_layer_sizes=(30,3))

# ensembles
# clf = AdaBoostRegressor(n_estimators=300, )

# clf = GradientBoostingRegressor(n_estimators=300)
# clf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3)

# clf = DecisionTreeClassifier(max_depth=6,min_samples_leaf=50,max_features=8)
# clf = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3)

# -----

# clf1 = RandomForestRegressor(n_estimators=320, n_jobs=-1, min_samples_leaf=3)
# clf2 = ExtraTreesRegressor(n_estimators=330, n_jobs=-1, min_samples_leaf=5)


# X_trainRandom = np.delete(X_train,[0],1)
# X_testRandom = np.delete(X_test,[0],1)

# X_trainExtra = np.delete(X_train,[0,2],1)
# X_testExtra = np.delete(X_test,[0,2],1)

# clf1.fit(X_trainRandom, Y_train)
# clf2.fit(X_trainExtra, Y_train)

# # training prediction
# print("training prediction")
# Ypred1 = clf1.predict(X_trainRandom)
# Ypred2 = clf2.predict(X_trainExtra)
# finalpred=(Ypred1+Ypred2)/2

# auc = roc_auc_score(Y_train, finalpred)
# print("Training auc: " + str(auc))

# # testing prediction
# print("Testing prediction")
# Ypred1 = clf1.predict(X_testRandom)
# Ypred2 = clf2.predict(X_testExtra)
# finalpred=(Ypred1+Ypred2)/2

# auc = roc_auc_score(Y_test, finalpred)
# print("Testing auc: " + str(auc))



# -------

# X_train = np.delete(X_train,[0,2],1)
# X_test = np.delete(X_test,[0,2],1)
# clf = ExtraTreesRegressor(n_estimators=330, n_jobs=-1, min_samples_leaf=5)

# X_train = np.delete(X_train,[0],1)
# X_test = np.delete(X_test,[0],1)
clf = RandomForestRegressor(n_estimators=320, n_jobs=-1, max_features=5, min_samples_leaf=3)

clf.fit(X_train, Y_train)
print(clf.feature_importances_)

# training prediction
print("training prediction")
Ypred = clf.predict(X_train)

mse = mean_squared_error(Y_train, Ypred)
print("training mse: " + str(mse))

auc = roc_auc_score(Y_train, Ypred)
print("Training auc: " + str(auc))

# testing prediction
print("Testing prediction")
Ypred = clf.predict(X_test)

mse = mean_squared_error(Y_test, Ypred)
print("Testing mse: " + str(mse))

auc = roc_auc_score(Y_test, Ypred)
print("Testing auc: " + str(auc))






