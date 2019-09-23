#!/usr/bin/env python
# coding: utf-8

# In[286]:


print ("\nAssignment 4 - 09/22/2019")
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
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from scipy import stats

# IMPORTING THE DATA #

# Import csv file
df = pd.read_csv('/Users/carolinacmleite/Desktop/housing2.csv',header=0)
# Display its shape (#rows and #columns)
print('The complete dataset has ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.\n')
print("------------\n")

# DS of Target Variable
print ("Descritive Statistics of Target Variable (MEDV)\n")
for x in range(-1,0):
    lastvar = df[df.columns[x]]
    print (lastvar.describe())
    print(" ")
print("------------\n")

# The count of the target is less than the total number of rows.
# It means we have some missing values in the target value. While we could choose an input method to infer these values,
# to make it simple (and also because we'll lose just 10% of the original data), I'll drop them.
df = df.dropna(subset=['MEDV'])
print('The dataset without the target missing values has ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.')

# Drop the target (MEDV) from the features array
X = df.drop('MEDV', axis=1).values
print('The features array has ' + str(X.shape[0]) + ' rows and ' + str(X.shape[1]) + ' columns.')
# Extract the target (MEDV) and move it to its own array
y = df['MEDV'].values
y = y.reshape(-1, 1)
print('The target array has ' + str(y.shape[0]) + ' rows.')
print("\n------------\n")

# DESCRIPTIVE STATISTICS #

# DS of Noise Variables (ATT1-13)
print ("Descriptive Statistics of Noise Variables (ATT1-13)\n")
for x in range(1, 14):
    print (df['ATT'+str(x)].describe())
    print(" ")
print("------------\n")

# DS of Original Variables
print ("Descriptive Statistics of Original Variables\n")
for x in range(-14, -1):
    lastvar = df[df.columns[x]]
    print (lastvar.describe())
    print(" ")
print("------------\n")

# DS of Target Variable
print ("Descriptive Statistics of Target Variable (MEDV)\n")
for x in range(-1,0):
    lastvar = df[df.columns[x]]
    print (lastvar.describe())
    print(" ")
print("------------\n")

# CORRELATION MATRIX (Only for the Original Variables + Target) #    

# Create an array X_orig with the original variables and the target
X_orig = df[df.columns[13:]]
# Compute Pearson correlation
corr = X_orig.corr()
# Create heat map with correlations calculated above
print ("Heat Map of Original Variables + Target\n")
plt.figure(figsize = (16,5))
sns.heatmap(corr, annot=True)
plt.show()
print("------------\n")

# SCATTERPLOT OF THE TWO FEATURES WITH THE HIGHEST CORRELATION WITH THE TARGET #
print ("Scatterplot of features RM and LSTAT (highest correlation with target) versus the target variable (MEDV)\n")
sns.set(style='whitegrid', context='notebook')
cols = ['RM','LSTAT','MEDV']
sns.pairplot(df[cols], height=2.5)
plt.show()
print("------------\n")

# SPLITTING TRAIN AND TEST SAMPLES + STANDARDIZATION #

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc_X = preprocessing.StandardScaler().fit(X_train)
sc_y = preprocessing.StandardScaler().fit(y_train)
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)

# 1st MODEL: LINEAR REGRESSION #

print ("1st Model: LINEAR REGRESSION\n")

# Setting the object with the model we're going to use
linreg = linear_model.LinearRegression()
# Fitting the model
linreg.fit(X_train, y_train)
# Printing coefficients and intercept
print ("Coefficients:")
print (linreg.coef_)
print ("\nIntercept: %.3f\n" %linreg.intercept_)
# Using the model to predict values of y for the training and test sets
y_train_pred = linreg.predict(X_train)
y_test_pred = linreg.predict(X_test)

# Residual plot
print ("Residual Plot\n")
plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-3, xmax=3, color='black', lw=2)
plt.show()
# Metrics: mean squared error
print ("Metrics\n")
print('MSE train: %.3f\nMSE test: %.3f' % (mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
# Metrics: R2
print('R2 train: %.3f\nR2 test: %.3f' % (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
print("\n------------\n")

# 2nd MODEL: RIDGE REGRESSION #

print ("2nd Model: RIDGE REGRESSION\n")

# Setup the array of alphas
rid_alpha = np.array([0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1])

# Computing scores over range of alphas
for a in rid_alpha:

    print('Alpha = ' + str(a))
    print(" ")
    # Creating a ridge regressor: ridreg
    ridreg = linear_model.Ridge(alpha=a, normalize=True)
    # Fitting the model
    ridreg.fit(X_train, y_train)
    # Printing coefficients and intercept
    print ("Coefficients:")
    print (ridreg.coef_)
    print ("\nIntercept: %.3f\n" %ridreg.intercept_)
    # Using the model to predict values of y for the training and test sets
    y_train_pred = ridreg.predict(X_train)
    y_test_pred = ridreg.predict(X_test)
    # Residual plot
    print ("Residual Plot\n")
    plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-3, xmax=3, color='black', lw=2)
    plt.show()
    # Metrics: MSE and R2
    print('MSE train: %.3f\nMSE test: %.3f' % (mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
    # Metrics: R2
    print('R2 train: %.3f\nR2 test: %.3f' % (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
    print(" ")

print("------------\n")
    
# 3rd MODEL: LASSO REGRESSION #

print ("3rd Model: LASSO REGRESSION\n")

# Setup the array of alphas
las_alpha = np.array([0.00001, 0.0001, 0.001, 0.0025, 0.005, 0.01, 0.03])

# Computing scores over range of alphas
for a in las_alpha:

    print('Alpha = ' + str(a))
    print(" ")
    # Creating a ridge regressor: ridreg
    lasreg = linear_model.Lasso(alpha=a, normalize=True)
    # Fitting the model
    lasreg.fit(X_train, y_train)
    # Printing coefficients and intercept
    print ("Coefficients:")
    print (lasreg.coef_)
    print ("\nIntercept: %.3f\n" %lasreg.intercept_)
    # Using the model to predict values of y for the training and test sets
    y_train_pred = lasreg.predict(X_train)
    y_test_pred = lasreg.predict(X_test)
    # Residual plot
    print ("Residual Plot\n")
    plt.scatter(y_train_pred, y_train_pred - y_train[:,0], c='steelblue', marker='o', edgecolor='white', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test[:,0], c='limegreen', marker='s', edgecolor='white', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-3, xmax=3, color='black', lw=2)
    plt.show()
    # Metrics: MSE and R2
    print('MSE train: %.3f\nMSE test: %.3f' % (mean_squared_error(y_train[:,0],y_train_pred),mean_squared_error(y_test[:,0],y_test_pred)))
    # Metrics: R2
    print('R2 train: %.3f\nR2 test: %.3f' % (r2_score(y_train[:,0], y_train_pred),r2_score(y_test[:,0], y_test_pred)))
    print(" ")

print("------------\n")
print('Correlation of Noise Variables (ATT) with the Target Variable (MEDV)\n')
ind_var_noise = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
for i in ind_var_noise:
    print('ATT'+ str(i) + ': ' + str(df['ATT'+str(i)].corr(df['MEDV'])))
 
print("\n------------\n")
print("My name is Carolina Carvalho Manhaes Leite")
print("My NetID is: leite2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:




