from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

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


X_new_train = []
Y_new_train = []

# make 0 and 1 predictions more equal. Remove every other 0 prediction
skip = False
for c,y in enumerate(Y_train):
	if y == 0:
		if skip:
			# np.delete(Y_train, c)
			# np.delete(X_train, c)
			skip = False
		else:
			X_new_train.append(X_train[c])
			Y_new_train.append(Y_train[c])
			skip=True
	else:
		X_new_train.append(X_train[c])
		Y_new_train.append(Y_train[c])

X_new_train = np.array(X_new_train)
Y_new_train = np.array(Y_new_train)

scaler.fit(X_new_train)
X_new_train = scaler.transform(X_new_train)
X_test = scaler.transform(X_test)

# clf = MLPClassifier(hidden_layer_sizes=(20,3))
clf = LogisticRegression(penalty='l1')

clf.fit(X_new_train, Y_new_train)

# training prediction
print("training prediction")
train_pred = clf.predict(X_new_train)
prediction = pd.DataFrame(train_pred, columns=['train predictions']).to_csv('prediction.csv')

mse = mean_squared_error(Y_new_train, train_pred)
print("mse: " + str(mse))

print(confusion_matrix(Y_new_train,train_pred))
print(classification_report(Y_new_train,train_pred))

# testing prediction
print("Testing prediction")
test_pred = clf.predict(X_test)

mse = mean_squared_error(Y_test, test_pred)
print("mse: " + str(mse))

print(confusion_matrix(Y_test,test_pred))
print(classification_report(Y_test,test_pred))







