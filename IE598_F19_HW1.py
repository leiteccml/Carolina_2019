#!/usr/bin/env python
# coding: utf-8

# In[17]:


from sklearn import datasets

iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target
print (X_iris.shape, y_iris.shape)
    # There was a syntax error before. The code was written as "print X_iris.shape, y_shape", but we have to
    # put it between parenthesis.
print (X_iris[0], y_iris[0])
    # There was a syntax error before. The code was written as "print X_iris.shape, y_shape", but we have to
    # put it between parenthesis.
    
from sklearn.model_selection import train_test_split
    # There was a syntax error before. The code was written as "from sklearn.cross_validation ...", but the
    # submodule "cross_validation" was renamed to "model_selection".
from sklearn import preprocessing
# Get dataset with only the first two attributes
X, y = X_iris[:, :2], y_iris
# Split the dataset into a training and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
print (X_train.shape, y_train.shape)
    # There was a syntax error before. The code was written as "print X_iris.shape, y_shape", but we have to
    # put it between parenthesis.
# Standardize the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt
colors = ['red', 'greenyellow', 'blue']
for i in range(len(colors)):
    # There was a syntax error before. The code was written as "for i in xrange".
    # There is no "xrange" in Python 3.
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

print("")
print("My name is: Carolina Carvalho Manhaes Leite")
print("My NetID is: leite2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
print("")


# In[ ]:




