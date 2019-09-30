#!/usr/bin/env python
# coding: utf-8

# In[62]:


print ("\nAssignment 5 - 09/26/2019")
print("Name: Carolina Carvalho Manhaes Leite")
print("NetID: leite2")
print("\n------------\n")

# IMPORTING LIBRARIES #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from time import process_time

# SETTINGS #

plt.clf()

# IMPORTING THE DATA #

# Import csv file
df = pd.read_csv('/Users/carolinacmleite/Documents/01 - Documents/04 - Academic/03 - Master/02 - Financial Engineering (UIUC)/01 - Fall 2019/IE598 - Machine Learning in Fin Lab/03 - Assignments/IE598_F19_HW5_Data_09292019.csv',header=0)
# Display its shape (#rows and #columns)
print('The original dataset has ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.\n')
print("------------\n")

# DS of Target Variable
print ("Descriptive Statistics of Target Variable (Adj_Close)\n")
for x in range(-1,0):
    lastvar = df[df.columns[x]]
    print (lastvar.describe())
    print(" ")
    #plt.boxplot(lastvar)
    #plt.show()
    #plt.clf()
    #print("--\n")
print("------------\n")

# The count of the target is less than the total number of rows.
# It means we have some missing values in the target value. While we could choose an input method to infer these values but,
# to make it simple, I'll drop them.
df = df.dropna(subset=['Adj_Close'])
print('The dataset without the target missing values has ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.')

# Drop the 1st column
df = df.iloc[:, 1:]
print('The dataset without the first column has ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.')
print("\n------------\n")

# Drop the target (Adj_Close) from the features array
X = df.drop('Adj_Close', axis=1).values
print('The features array has ' + str(X.shape[0]) + ' rows and ' + str(X.shape[1]) + ' columns.')
# Extract the target (Adj_Close) and move it to its own array
y = df['Adj_Close'].values
y = y.reshape(-1, 1)
print('The target array has ' + str(y.shape[0]) + ' rows.')
print("\n------------\n")

# DESCRIPTIVE STATISTICS #

# DS of Feature Variables
#print ("Descriptive Statistics of Original Variables\n")
#for x in range(0, 30):
    #lastvar = df[df.columns[x]]
    #print (lastvar.describe())
    #print(" ")
    #plt.boxplot(lastvar)
    #plt.show()
    #plt.clf()
    #print("--\n")
#print("------------\n")

# CORRELATION MATRIX #    

# Compute Pearson correlation
corr = df.corr()
# Create heat map with correlations calculated above
print ("Heat Map of Features + Target Correlation Matrix\n")
sns.heatmap(corr)
plt.show()
plt.clf()
print ("\nCorrelations with the target variable\n")
print(np.transpose(corr[30:]))
print("\n------------\n")

X_axis_corr = np.arange(1, 31)
y_axis_corr = -1*np.transpose(corr[30:])
y_axis_corr = y_axis_corr[1:31]
axes = plt.gca()
axes.set_ylim([0.5,1])
plt.scatter(X_axis_corr, y_axis_corr, alpha=0.5)
plt.title('Scatter plot: Correlation of features with target variable')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.clf()
print(" ")

X_max_corr = df['SVENF15'].values
X_min_corr = df['SVENF30'].values
plt.scatter(y, X_max_corr, alpha=0.5)
plt.title('Scatter plot: feature with maximum correlation x target')
plt.xlabel('Adj_Close (Target)')
plt.ylabel('SVENF15')
plt.show()
plt.clf()
print(" ")
plt.scatter(y, X_min_corr, alpha=0.5)
plt.title('Scatter plot: feature with minimum correlation x target')
plt.xlabel('Adj_Close (Target)')
plt.ylabel('SVENF30')
plt.show()
plt.clf()
print("\n------------\n")

# SPLITTING TRAIN AND TEST SAMPLES + STANDARDIZATION #

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)
sc_X = preprocessing.StandardScaler().fit(X_train)
sc_y = preprocessing.StandardScaler().fit(y_train)
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)

# PCA #

# Setting the object with the PCA (with all principal components)
pca = PCA()
X_train_PCA = pca.fit_transform(X_train)
X_test_PCA = pca.transform(X_test)

# Printing table and charts with the explained variance (individual and cumulative)
variance_pca_full = np.array(pca.explained_variance_ratio_)
print("The explained variance for each of the components/features is (in order):\n")
print(variance_pca_full)
print("\nGraphically:\n")
y_full = np.arange(len(variance_pca_full))
plt.bar(y_full, variance_pca_full)
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.show()
print(" ")
print("The cumulative explained variance is (in order):\n")
print(np.cumsum(pca.explained_variance_ratio_))
print("\nGraphically:\n")
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()
print("\n------------\n")

# Setting the object with the PCA (with 3 components)
pca3 = PCA(n_components=3)
X_train_PCA3 = pca3.fit_transform(X_train)
X_test_PCA3 = pca3.transform(X_test)

# Printing table and charts with the explained variance (individual and cumulative)
variance_pca_3 = np.array(pca3.explained_variance_ratio_)
print("The cumulative explained variance for those 3 components is (in order):\n")
print(np.cumsum(pca3.explained_variance_ratio_))
sum_variance = np.cumsum(pca3.explained_variance_ratio_)
print("\nIt means that, together, they explain %.3f of all variance." % sum_variance[2])
print("\n------------\n")

# 1st MODEL: LINEAR REGRESSION WITH ALL ATTRIBUTES #

print ("1st Model: LINEAR REGRESSION with all attributes")

# Setting the object with the model we're going to use
linreg_all = linear_model.SGDRegressor(loss = 'squared_loss', penalty=None, random_state=42)
# Setting the start time
start = process_time()
# Fitting the model
linreg_all.fit(X_train, y_train.ravel())
# Setting the end time
end = process_time()
# Calculating processing time
proc_time = end - start
# Using the model to predict values of y for the training and test sets
y_train_pred_all = linreg_all.predict(X_train)
y_test_pred_all = linreg_all.predict(X_test)

# Metrics: mean squared error
print ("Metrics:\n")
print('MSE train: %.3f\nMSE test: %.3f' % (mean_squared_error(y_train,y_train_pred_all),mean_squared_error(y_test,y_test_pred_all)))
# Metrics: R2
print('R2 train: %.3f\nR2 test: %.3f' % (r2_score(y_train, y_train_pred_all),r2_score(y_test, y_test_pred_all)))

# Processing time
print ('\nThe processing time was: %.3f' % proc_time)
print("\n------------\n")

# 2nd MODEL: LINEAR REGRESSION WITH 3 PRINCIPAL COMPONENTS #

print ("2nd Model: LINEAR REGRESSION with 3 principal components")

# Setting the object with the model we're going to use
linreg_3 = linear_model.SGDRegressor(loss = 'squared_loss', penalty=None, random_state=42)

# Setting the start time
start = process_time()
# Fitting the model
linreg_3.fit(X_train_PCA3, y_train.ravel())
# Setting the end time
end = process_time()
# Calculating processing time
proc_time = end - start
# Using the model to predict values of y for the training and test sets
y_train_pred_3 = linreg_3.predict(X_train_PCA3)
y_test_pred_3 = linreg_3.predict(X_test_PCA3)

# Metrics: mean squared error
print ("Metrics:\n")
print('MSE train: %.3f\nMSE test: %.3f' % (mean_squared_error(y_train,y_train_pred_3),mean_squared_error(y_test,y_test_pred_3)))
# Metrics: R2
print('R2 train: %.3f\nR2 test: %.3f' % (r2_score(y_train, y_train_pred_3),r2_score(y_test, y_test_pred_3)))

# Processing time
print ('\nThe processing time was: %.3f' % proc_time)
print("\n------------\n")

# 3rd MODEL: SVR WITH WITH ALL ATTRIBUTES #

print ("3rd Model: SVR with all attributes")

# Setting the object with the model we're going to use
svm_all = SVR(kernel = 'linear', gamma='scale', C=1.0)
# Setting the start time
start = process_time()
# Fitting the model
svm_all.fit(X_train, y_train.ravel())
# Setting the end time
end = process_time()
# Calculating processing time
proc_time = end - start
# Using the model to predict values of y for the training and test sets
y_train_pred_all = svm_all.predict(X_train)
y_test_pred_all = svm_all.predict(X_test)

# Metrics: mean squared error
print ("Metrics:\n")
print('MSE train: %.3f\nMSE test: %.3f' % (mean_squared_error(y_train,y_train_pred_all),mean_squared_error(y_test,y_test_pred_all)))
# Metrics: R2
print('R2 train: %.3f\nR2 test: %.3f' % (r2_score(y_train, y_train_pred_all),r2_score(y_test, y_test_pred_all)))

# Processing time
print ('\nThe processing time was: %.3f' % proc_time)
print("\n------------\n")

# 4th MODEL: SVR WITH WITH 3 PRINCIPAL COMPONENTS #

print ("4th Model: SVR with 3 principal components")

# Setting the object with the model we're going to use
svm_3 = SVR(kernel = 'linear', gamma='scale', C=1.0)
# Setting the start time
start = process_time()
# Fitting the model
svm_3.fit(X_train_PCA3, y_train.ravel())
# Setting the end time
end = process_time()
# Calculating processing time
proc_time = end - start
# Using the model to predict values of y for the training and test sets
y_train_pred_3 = svm_3.predict(X_train_PCA3)
y_test_pred_3 = svm_3.predict(X_test_PCA3)

# Metrics: mean squared error
print ("Metrics:\n")
print('MSE train: %.3f\nMSE test: %.3f' % (mean_squared_error(y_train,y_train_pred_3),mean_squared_error(y_test,y_test_pred_3)))
# Metrics: R2
print('R2 train: %.3f\nR2 test: %.3f' % (r2_score(y_train, y_train_pred_3),r2_score(y_test, y_test_pred_3)))

# Processing time
print ('\nThe processing time was: %.3f' % proc_time)
print("\n------------\n")

print("My name is Carolina Carvalho Manhaes Leite")
print("My NetID is: leite2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:




