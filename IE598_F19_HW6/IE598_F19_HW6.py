#!/usr/bin/env python
# coding: utf-8

# In[81]:


print ("\nAssignment 6 - 10/06/2019")
print("Name: Carolina Carvalho Manhaes Leite")
print("NetID: leite2")
print("\n------------\n")

# IMPORTING LIBRARIES #

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from time import process_time

# IMPORTING THE DATA #

# Import csv file
df = pd.read_csv('/Users/carolinacmleite/Documents/01 - Documents/04 - Academic/03 - Master/02 - Financial Engineering (UIUC)/01 - Fall 2019/IE598 - Machine Learning in Fin Lab/03 - Assignments/ccdefault.csv',header=0)
# Display its shape (#rows and #columns)
print('The dataset has ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.\n')
print("------------\n")

# DS of Target Variable
print ("Descriptive Statistics of Target Variable (DEFAULT)\n")
for x in range(-1,0):
    lastvar = df[df.columns[x]]
    print (lastvar.describe())
    print(" ")
print("------------\n")

# PREPROCESSING AND MANIPULATING THE DATA #

# Drop the 1st column (ID)
df = df.iloc[:, 1:]
# Drop the target (DEFAULT) from the features array
X = df.drop('DEFAULT', axis=1)
print('The features array has ' + str(X.shape[0]) + ' rows and ' + str(X.shape[1]) + ' columns.')
# Extract the target (DEFAULT) and move it to its own array
y = df['DEFAULT'].values
print('The target array has ' + str(y.shape[0]) + ' rows.')
print("\n------------\n")

# Pipelines for the preprocessing of numeric and categorical variables
num_var = ['LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
steps = [('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())]
num_transf = Pipeline(steps)
cat_var = ['SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
steps = [('onehot', OneHotEncoder(handle_unknown='ignore'))]
cat_transf = Pipeline(steps)

# Apply the transformations
preproc = ColumnTransformer(transformers=[('num', num_transf, num_var),('cat', cat_transf, cat_var)])

# Appending Decision Tree Classifier to the Preprocessing pipeline

# Method 1) We're changing manually (with a loop) the random_state from the DecisionTreeClassifier from 1 to 10
start = process_time()
index = np.arange(1, 11)
acc_manual_train = np.empty(len(index)+1)
acc_manual_test = np.empty(len(index)+1)
for i in index:
    # Splitting train and test data (stratifying the target value y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i, stratify=y)
    steps=[('preproc',preproc),('classifier', DecisionTreeClassifier(max_depth=7,random_state=1))]
    clf = Pipeline(steps)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    acc_manual_train[i] = accuracy_score(y_train, y_pred_train)
    acc_manual_test[i] = accuracy_score(y_test, y_pred_test)
    
print("RANDOM TRAIN_TEST SPLIT METHOD\n")
print("Out-of-sample accuracy scores: ")
acc_manual_train = np.delete(acc_manual_train, np.s_[0:1], axis=0)
acc_manual_test = np.delete(acc_manual_test, np.s_[0:1], axis=0)
print(acc_manual_test)
print("")
print("Out-of-sample accuracy scores (mean): %.3f " % np.mean(acc_manual_test))
print("Out-of-sample accuracy scores (std_dev): %.3f " % np.std(acc_manual_test))
print("")
print("In-sample accuracy scores: ")
print(acc_manual_train)
print("")
print("In-sample accuracy scores (mean): %.3f " % np.mean(acc_manual_train))
print("In-sample accuracy scores (std_dev): %.3f " % np.std(acc_manual_train))
end = process_time()
proc_time = end - start
print("Process time: %.3f " % proc_time)
print("\n------------\n")
    
# Method 2) We're using K-Fold CV (with 10 folds)
start = process_time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)
steps=[('preproc',preproc),('classifier', DecisionTreeClassifier(max_depth=7,random_state=1))]
clf2 = Pipeline(steps)
acc_kfold = cross_val_score(clf2, X_train, y_train, cv=10)
print("STRATIFIED K-F CV METHOD\n")
print("Accuracy scores: ")
print(acc_kfold)
print("")
print("Accuracy scores (mean): %.3f " % np.mean(acc_kfold))
print("Accuracy scores (std_dev): %.3f " % np.std(acc_kfold))
end = process_time()
proc_time = end - start
print("Process time: %.3f " % proc_time)
print("\n------------\n")

print("My name is: Carolina Carvalho Manhaes Leite")
print("My NetID is: leite2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
print("")


# In[ ]:




