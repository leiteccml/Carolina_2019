#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Our first machine learning model
#Garreta and Moncecchi pp 10-20
#uses Iris database and SGD classifier
import sklearn
print( 'The scikit learn version is {}.'.format(sklearn.__version__))
print("")

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

from sklearn.linear_model import SGDClassifier
    # There was an error before. The code was written as "sklearn.linear_modelsklearn._model".
    # The correct code would be "sklearn.linear_model".
clf = SGDClassifier(random_state=1000000, max_iter =1000000)
    # According to the SGDClassifier's documentation, the function has an inner process that shuffles/splits
    # the data, using a random seed defined by the variable "random_state" (one of the functions' parameters).
    # When the user doesn’t specify a fixed seed, the parameter is automatically set to “None”, implying that
    # at each and every time we run the code we can get different seeds and, thus, different results.
    # The original code used the function's default parameters, giving different results at each processing.
    # I set a fixed "random_state" (for example: 33) and was able to consistently get the same results
    # in multiple runs. Still, I wasn't able to get the same intercept/coefficients the book did. If my
    # understanding is correct, I wouldn’t probably be able to replicate them, once the "random_state" for 
    # the example was initialized as “None", being, in this case, random and unknown.

clf.fit(X_train, y_train)
print (clf.coef_)
print (clf.intercept_)
    # There was a syntax error before. Both "print" were missing the parenthesis.

import numpy as np
import matplotlib
    # These packages were not being imported in the original code. They will be needed, though.   
x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
xs = np.arange(x_min, x_max, 0.5)
    # The code used "np" (NumPy package) without importing it. I made sure it was imported before running this
    # part of the code. Besides, there is no such thing as "arrange"; it's "arange".
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(10, 6)
for i in [0, 1, 2]:
    axes[i].set_aspect('equal')
    axes[i].set_title('Class '+ str(i) + ' versus the rest')
    axes[i].set_xlabel('Sepal length')
    axes[i].set_ylabel('Sepal width')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    matplotlib.pyplot.sca(axes[i])
        # There was a syntax error before. The code was written as "sca", but that is a function defined
        # within a package, the Matplotlib.pyplot. We have to import it at the beginning of the code and
        # declare it before the name of the function. The new code will be "matplotlib.pyplot.sca(axes[i])".
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.prism)
    ys = (-clf.intercept_[i] - xs * clf.coef_[i, 0])/clf.coef_[i, 1]
        # We have to be careful with variables' names, because python is case sensitive.
        # The variable originally created was "xs" and not "Xs".
    plt.plot(xs, ys)
    
print (clf.predict(scaler.transform([[4.7, 3.1]])))
    # There was a syntax error before. Outer parenthesis were missing.
print (clf.decision_function(scaler.transform([[4.7, 3.1]])))

from sklearn import metrics
y_train_pred = clf.predict(X_train)
print (metrics.accuracy_score(y_train, y_train_pred))
    # There was a syntax error before. Outer parenthesis were missing.
y_pred = clf.predict(X_test)
print (metrics.accuracy_score(y_test, y_pred))
print ("")
    # There was a syntax error before. Outer parenthesis were missing.
print (metrics.classification_report(y_test, y_pred, target_names=iris.target_names))
print (metrics.confusion_matrix(y_test, y_pred))

print("")
print("My name is: Carolina Carvalho Manhaes Leite")
print("My NetID is: leite2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
print("")


# In[3]:


import pandas as pd
import numpy as np

print ("This is the OPTIONAL homework from Lesson 1 (The Treasury Squeeze Test)")
print ("")

# Import csv file
df = pd.read_csv('/Users/carolinacmleite/Desktop/Treasury_squeeze_test_DS1.csv',header=0)
# Display its shape (#rows and #columns)
print(df.shape)
# Drop the 1st and 2nd columns (rowindex and contract)
df = df.iloc[:, 2:]
print(df.shape)
# Extract the target and move it to its own array
y = df.pop('squeeze').values
# Convert boolean to int
y = y.astype(int)
print(y.shape)
# Drop the target column from the original data
X = df.iloc[:, :11]
print(X.shape)

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Split the dataset into a training and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
print (X_train.shape, y_train.shape)
# Standardize the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Model
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(random_state=1000000)
clf.fit(X_train, y_train)
# Printing the coefficients and the intercept
print (clf.coef_)
print (clf.intercept_)

# Metrics
from sklearn import metrics
y_train_pred = clf.predict(X_train)
print (metrics.accuracy_score(y_train, y_train_pred))
y_pred = clf.predict(X_test)
print (metrics.accuracy_score(y_test, y_pred))
print ("")
print (metrics.classification_report(y_test, y_pred))
print (metrics.confusion_matrix(y_test, y_pred))

print("")
print("My name is: Carolina Carvalho Manhaes Leite")
print("My NetID is: leite2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
print("")

