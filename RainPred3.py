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

# clf1 = RandomForestRegressor(n_estimators=300, n_jobs=-1, max_features=10, min_samples_leaf=4)
# clf2 = ExtraTreesRegressor(n_estimators=330, n_jobs=-1, max_features=10, min_samples_leaf=4) # was 3

# X_train = np.delete(X_train,[0],1)
# X_test = np.delete(X_test,[0],1)

# clf1.fit(X_train, Y_train)
# clf2.fit(X_train, Y_train)

# # training prediction
# print("training prediction")
# Ypred1 = clf1.predict(X_train)
# Ypred2 = clf2.predict(X_train)
# finalpred=(Ypred1+Ypred2)/2

# auc = roc_auc_score(Y_train, finalpred)
# print("Training auc: " + str(auc))

# # testing prediction
# print("Testing prediction")

# preds = []

# Ypred1 = clf1.predict(X_test)
# Ypred2 = clf2.predict(X_test)

# for i in range(len(Ypred1)):
# 	pred1 = Ypred1[i]
# 	pred2 = Ypred2[i]
# 	if pred1 > .5 and pred2 > .5:
# 		if pred1 > pred2:
# 			Ypred2[i] = pred1
# 	elif pred1 < .5 and pred2 < .5:
# 		if pred1 < pred2:
# 			Ypred2[i] = pred1
# 	else:
# 		Ypred2[i] = (Ypred1[i]+Ypred2[i])/2

# auc = roc_auc_score(Y_test, Ypred2)
# print("Testing auc: " + str(auc))



# -------

# need to get to .766 for top 10
# need to gtet to .77 to win need to get to 

X_train = np.delete(X_train,[0],1) # 2 and 11
X_test = np.delete(X_test,[0,],1)
clf = ExtraTreesRegressor(n_estimators=700, n_jobs=-1, max_features=10, min_samples_leaf=4, max_depth=30) # was 3

# clfe = DecisionTreeRegressor(max_depth=6,min_samples_leaf=50,max_features=10)
# clf = GradientBoostingRegressor(n_estimators=400, max_features=10, min_samples_leaf=50)

# X_train = np.delete(X_train,[0],1)
# X_test = np.delete(X_test,[0],1)
# clf = RandomForestRegressor(n_estimators=360, n_jobs=-1, max_features=9, min_samples_leaf=4)

clf.fit(X_train, Y_train)
# print(clf.feature_importances_)

# training prediction
print("training prediction")
Ypred = clf.predict(X_train)

auc = roc_auc_score(Y_train, Ypred)
print("Training auc: " + str(auc))

# testing prediction
print("Testing prediction")
Ypred = clf.predict(X_test)

auc = roc_auc_score(Y_test, Ypred)
print("Testing auc: " + str(auc))






