import numpy as np
import pandas as pd
import matplotlib as plt
import math

veriler = pd.read_csv('/home/ikve/Downloads/train.csv')
yeni_veri = veriler[['Pclass','Sex', 'Age', 'SibSp', 'Fare', 'Embarked', 'Parch']]
y = veriler[['Survived']]

# print(yeni_veri.isnull().sum())
yeni_veri=pd.get_dummies(yeni_veri, dummy_na=True)
yeni_veri = yeni_veri.drop('Sex_nan', axis=1)

from sklearn.preprocessing import Imputer
imp = Imputer()
imp_age = imp.fit_transform(yeni_veri[['Age']].values)
df = pd.DataFrame(imp_age)
yeni_veri[['Age']] = df

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
cabin = veriler[['Cabin']].fillna('aaaaaa').values
cabin = lb.fit_transform(cabin)
yeni_veri[['Cabin']]=pd.DataFrame(cabin)

# age = yeni_veri.iloc[:, 2].values
# age = age.reshape(891,1)
#
# from sklearn.preprocessing import Imputer
# imp = Imputer()
# age  = imp.fit_transform(age)

# print(yeni_veri[['Sex']])
# print(pd.get_dummies(yeni_veri[['Sex']]))

# from sklearn.preprocessing import LabelEncoder
# embarked = yeni_veri.iloc[:, -1].values.astype(str)
# le = LabelEncoder()
# embarked = le.fit_transform(embarked)
#
# sex = yeni_veri.iloc[:, 1].values.astype(str)
# le = LabelEncoder()
# sex = le.fit_transform(sex)
#
#
# df = pd.DataFrame(age, columns=['Age'])
# df2 = pd.DataFrame(embarked, columns=['Embarked'])
# df3 = pd.DataFrame(sex, columns=['Sex'])
# yeni_veri = veriler[['Pclass', 'SibSp', 'Fare']]
#
# yeni_veri = pd.concat([df, df2, df3, yeni_veri], axis=1)
#

from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test)=train_test_split(yeni_veri, y)

# from sklearn.linear_model import Perceptron
# perc = Perceptron(max_iter=1000000)
# perc.fit(x_train, y_train)
# pred = perc.predict(x_test)
#
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# matrix = confusion_matrix(y_test, pred)
# print(matrix)
#
# from sklearn.linear_model import LogisticRegression
# logr = LogisticRegression(C=0.8)
# logr.fit(x_train, y_train)
# pred = logr.predict(x_test)
# matrix = confusion_matrix(y_test, pred)
# print(matrix)
# print(accuracy_score(y_test,pred))

from sklearn.svm import SVC

svm = SVC(C=0.75, kernel='poly')
svm.fit(x_train, y_train)
pred = svm.predict(x_test)
matrix = confusion_matrix(y_test, pred)
print(matrix)
print(accuracy_score(y_test, pred))

# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(x_train, y_train)
# pred = knn.predict(x_test)
# matrix = confusion_matrix(y_test, pred)
# print(matrix)
#
# from  sklearn import tree
#
# tree = tree.DecisionTreeClassifier(criterion='entropy')
# tree.fit(x_train, y_train)
# pred = tree.predict(x_test)
# matrix = confusion_matrix(y_test, pred)
# print(matrix)



