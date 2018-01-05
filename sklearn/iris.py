#!/usr/bin/env python
# coding=utf-8

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

iris_x = iris.data
iris_y = iris.target

print iris_x.shape
print iris_y.shape

X_train, X_test, Y_train, Y_test = train_test_split(iris_x, iris_y, test_size=0.4)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)

print knn.predict(X_test)
print knn.score(X_test, Y_test)

