import numpy as np
import matplotlib as plt
from sklearn.datasets import load_iris

data=load_iris()
X=data['data']
Y=data['target']

from  sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X,Y, random_state=0, test_size=0.33)

from  sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

# from sklearn.linear_model import Perceptron
# perc = Perceptron(max_iter=1000000,eta0=0.001)
# perc.fit(x_train, y_train)
# predict = perc.predict(x_test)

# from sklearn.linear_model import SGDClassifier
# adaline = SGDClassifier(max_iter=100000)
# adaline.fit(x_train, y_train)
# predict = adaline.predict(x_test)

# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(C=10000.0)
# lr.fit(x_train, y_train)
# predict = lr.predict(x_test)


# from sklearn.svm import SVC
# svm = SVC( C=10, gamma = 0.5)
# svm.fit(x_train, y_train)
# predict = svm.predict(x_test)

# from sklearn.tree import DecisionTreeClassifier
# dt = DecisionTreeClassifier(criterion='gini')
# dt.fit(x_train, y_train)
# predict = dt.predict(x_test)

# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
# rfc.fit(x_train, y_train)
# predict = rfc.predict(x_test)

# from sklearn.metrics import confusion_matrix
# from  sklearn.metrics import accuracy_score
# print(accuracy_score(y_test,predict))
# print(confusion_matrix(y_test,predict))

print(Y)
