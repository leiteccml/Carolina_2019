#!/usr/bin/env python
# coding: utf-8

# In[134]:


import pandas as pd
import numpy as np

print ("Assignment 2 - 09/08/2019")
print ("")

# IMPORTING AND MANIPULATING THE DATA #

print ("# DATA SCREENING")
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

#print(X)

# SPLITTING TRAINING AND TESTING SETS #

from sklearn.model_selection import train_test_split

# Split the dataset into a training and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print (X_train.shape, y_train.shape)
print ("")
# There's no need to standardize the features since they're all binary, with values 0 and 1

# 1ST MODEL: k-NN CLASSIFIER #
# Source of code general structures: DataCamp, Supervised Learning with scikit-learn, Chapter 1: Classification
print ("# 1ST MODEL: k-NN CLASSIFIER")
print ("")

# Importing KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 10)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

print ("k", "Accuracy (Train)", "Accuracy (Test)")
# Looping over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    # Computing accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    # Computing accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)
    # Print values in a table
    print(k, "      ", round(train_accuracy[i],3), "        ", round(test_accuracy[i],3))

max_acc_knn = max(test_accuracy)
# Array of indices to be added as column
indexes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) 
# Adding column of indices to original array (test_accuracy)
test_acc_index = np.column_stack((indexes, test_accuracy))
print ("")
print("Values of k and maximum accuracy in the test sample are: ")
max_knn = print(test_acc_index[np.argsort(test_acc_index[:, 1])][-1]) # Source: https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
print ("")

# Generating plot
import matplotlib.pyplot as plt

plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

# 2ND MODEL: DECISION TREE CLASSIFIER #
# Source of code general structures: DataCamp, Machine Learning with Tree-Based Models in Python,
# Chapter 1: Classification and Regression Trees
print ("")
print ("# 2ND MODEL: DECISION TREE CLASSIFIER")
print ("")

# Visualizing countplots 2x2
# Sources: DataCamp, Supervised Learning with scikit-learn, Chapter 1: Classification (Visual EDA)
#          https://seaborn.pydata.org/generated/seaborn.countplot.html
import seaborn as sns

# Import csv file again (we want to keep all variables and headers)
df_plot = pd.read_csv('/Users/carolinacmleite/Desktop/Treasury_squeeze_test_DS1.csv',header=0)
# Drop the 1st and 2nd columns (rowindex and contract)
df_plot = df_plot.iloc[:, 2:]

# Creating countplots
plt.figure()
sns.countplot(x='price_crossing', hue='squeeze', data=df_plot, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()
plt.figure()
sns.countplot(x='price_distortion', hue='squeeze', data=df_plot, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()
plt.figure()
sns.countplot(x='roll_start', hue='squeeze', data=df_plot, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()
plt.figure()
sns.countplot(x='roll_heart', hue='squeeze', data=df_plot, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()
plt.figure()
sns.countplot(x='near_minus_next', hue='squeeze', data=df_plot, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()
plt.figure()
sns.countplot(x='ctd_last_first', hue='squeeze', data=df_plot, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()
plt.figure()
sns.countplot(x='ctd1_percent', hue='squeeze', data=df_plot, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()
plt.figure()
sns.countplot(x='delivery_cost', hue='squeeze', data=df_plot, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()
plt.figure()
sns.countplot(x='delivery_ratio', hue='squeeze', data=df_plot, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()
print ("")

# Importing DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier
# Importing accuracy_score
from sklearn.metrics import accuracy_score

# Setup arrays to store train and test accuracies
depth = np.arange(1, 10)
test_accuracy_dt = np.empty(len(depth))

print ("Max_Depth", "Accuracy (Test)")
# Looping over different values of max_depth
for i, k in enumerate(depth):
    # Instantiating a DecisionTreeClassifier 'dt' with a maximum depth of k
    dt = DecisionTreeClassifier(max_depth=k, random_state=1)
    # Fitting the classifier to the training data
    dt.fit(X_train,y_train)
    # Predicting test set labels
    y_pred = dt.predict(X_test)
    # Computing accuracy on the testing set
    test_accuracy_dt[i] = accuracy_score(y_test, y_pred)
    # Print values in a table
    print("   ", k, "        ", round(test_accuracy_dt[i],3))

max_acc_dt = max(test_accuracy_dt)
    
print ("")
    
# Generating plot
import matplotlib.pyplot as plt

plt.title('Decision Tree: Varying Value of Max_Depth')
plt.plot(neighbors, test_accuracy_dt, label = 'Testing Accuracy')
plt.legend()
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')
plt.show()

# Instantiating a DecisionTreeClassifier 'dt' with a maximum depth of 3
# It's the depth >= 1 that leads to the maximum accuracy
dt = DecisionTreeClassifier(max_depth=3, random_state=1)
# Fitting the classifier to the training data
dt.fit(X_train,y_train)

# Creating a file that can be used to construct the tree flow chart
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
# After its creation, the text in dt.dot can be uploaded to http://webgraphviz.com, creating the chart
from sklearn import tree
dotfile = open("dt.dot", 'w')
tree.export_graphviz(dt, out_file=dotfile)
dotfile.close()

print("")
print("My name is: Carolina Carvalho Manhaes Leite")
print("My NetID is: leite2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
print("")


# In[ ]:




