from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

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
# clf = DecisionTreeClassifier()

# ensembles
# clf = AdaBoostClassifier(n_estimators=82)
# clf = GradientBoostingClassifier(n_estimators=82, learning_rate=1, max_depth=1, random_state=1)

clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=50, max_features=8)
# clf = DecisionTreeClassifier(max_depth=6,min_samples_leaf=50,max_features=8)
# clf = ExtraTreesClassifier(n_estimators=100, max_features=8)
clf.fit(X_train, Y_train)

# training prediction
print("training prediction")
train_pred = clf.predict(X_train)

print(classification_report(Y_train,train_pred))

# testing prediction
print("Testing prediction")
test_pred = clf.predict(X_test)

	# mse = mean_squared_error(Y_test, test_pred)
	# if mse < smallest_mse:
	# 	smallest_mse = mse
	# 	d1 = d
	# 	print("MSE: " + str(mse))
	# 	print("max depth: " + str(d1))
	# print("done")

print(classification_report(Y_test,test_pred))







