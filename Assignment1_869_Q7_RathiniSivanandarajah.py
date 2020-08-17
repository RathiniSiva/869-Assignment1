#!/usr/bin/env python
# coding: utf-8

# In[152]:


# [Rathini, Sivanandarajah]
# [Student number: 20220479]
# [MMA]
# [2021W]
# [MMA 869]
# [July 14,2020]


# Answer to Question [7], Part [1 and 2]
# Building 3 models for OJ.csv dataset. Model 1 is Decsion Tree, Model 2 is XGBoost and Model 3 is Random Forest.


# In[153]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import silhouette_score, silhouette_samples
import sklearn.metrics

import itertools
import scipy
import xgboost as xgb
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[154]:


#to find out current directory where python file is saved
import os
os.getcwd()


# In[155]:


#Reading in Data

df = pd.read_csv("OJ.csv")

df.rename( columns={'Unnamed: 0':'ID'}, inplace=True )


# In[156]:


# Creating an ID column and setting target variable to 'Purchase'
Id_col = 'ID'
target_col = 'Purchase'
df.info()
df.head()


# In[159]:


## descriptive analysis
df.describe()


# # Data cleansing 

# In[160]:


# Changed target variable to a numerical variable, CH = 0 and MM = 1 - to prepare for binary classification models
# Changed Store7  to a numerical variable as well, Yes = 1, six = 6 and No = 0.
# This will be useful for models such as XGBoost

cleanup_nums = {"Purchase":     {"CH": 0, "MM": 1},
                "Store7": {"Yes": 1, "six": 6, "No": 0}}


# In[161]:


# reaplacing numerical values as assigned above

df.replace(cleanup_nums, inplace=True)
df.head()


# In[162]:


#check for imbalance in target variable after converting to binary
#CH = 0, MM = 1

df['Purchase'].value_counts()


#0    653
#1    417
#Name: Purchase, dtype: int64
#There are 61% CH Citrus Hill purchasers and 39% MM Minute Maid purchasers
#We do not need to balance our data in this case as we have good representation from both classes


# In[58]:


## correlation
df.corr()


# In[59]:


# Heatmap to show correlation between features and target variable

X = df.iloc[:,0:18]  #independent columns
y = df.iloc[:,-1]    #target column i.e price range
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(18,18))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[23]:


#Feature Importance
#Identifying most important features using ExtraTrees classifier

X = df.iloc[:,2:17]  #independent columns
y = df.iloc[:,1]    #target column i.e price range
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()


# In[21]:


#shows the first few rows of target variable
y.head() 


# In[22]:


# printing out all column names
data_top = df.head()

data_top


# In[60]:


from sklearn.model_selection import train_test_split

# droping some features and assigning features to X and target to y
X = df.drop([Id_col, target_col,"PctDiscCH","PctDiscMM","PriceCH","PriceMM"], axis=1)

y = df[target_col]


# # Splitting data into Train and Test

# In[61]:


#splitting data into train and test (80% train and 20% test)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[62]:


##check the shape of the training set 
X_train.shape


# In[63]:


## shape of the test size
X_val.shape


# In[64]:


## shape of test - target variable
y_val.shape


# In[65]:


X.info()
X.shape
X.head()

X_train.info()
X_train.shape
X_train.head()


# # Build Model 1 - DecisionTree classifier
# 

# In[136]:


# tuning our Decision Tree  classifier

from sklearn.tree import DecisionTreeClassifier 

clf2 = DecisionTreeClassifier(random_state=42, criterion="gini",min_samples_split= 4, min_samples_leaf=5, max_depth=50, max_leaf_nodes=10)
#this gave an accuracy of 0.8 which was the best I could get after trying different hyper parameters

clf2.fit(X_train, y_train)

pred_val2 = clf2.predict(X_val)

print('confusion matrix for Final DecisionTree model')
confusion_matrix(y_val, pred_val2)


# # Estimate Model Performance of DecisionTree 

# In[167]:


# performance metrics for tuned Decision Tree model


from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss

print("Accuracy for Decision Tree Model = {:.2f}".format(accuracy_score(y_val, pred_val2)))
print("Kappa = {:.2f}".format(cohen_kappa_score(y_val, pred_val2)))
print("F1 Score = {:.2f}".format(f1_score(y_val, pred_val2)))
print("Log Loss = {:.2f}".format(log_loss(y_val, pred_val2)))


# In[126]:


from sklearn.metrics import classification_report, f1_score

print("Accuracy for Decision Tree model = {:.2f}".format(accuracy_score(y_val, pred_val2)))


# # Model 2 - XGBOOST
# 
# 

# In[113]:


from sklearn.model_selection import train_test_split


# droping some features
#note X2, y2 as I dropped different combination of features for this model

X2 = df.drop([Id_col, target_col,"PctDiscCH","ListPriceDiff","PctDiscMM","PriceMM","PriceCH"], axis=1)

y2 = df[target_col]

from sklearn.model_selection import train_test_split

X_train2, X_val2, y_train2, y_val2 = train_test_split(X2, y2, test_size=0.2, random_state=42)


# In[114]:


#Train the XGboost Model for Classification
model1 = xgb.XGBClassifier()
model2 = xgb.XGBClassifier(random_state = 0,n_estimators=30, max_depth=30, learning_rate=0.02, subsample=0.3)

#model2 = xgb.XGBClassifier(n_estimators=30, max_depth=30, learning_rate=0.02, subsample=0.3)
#Accuracy for model 2: 84.11 ----use this hyperparameter

train_model1 = model1.fit(X_train2, y_train2)
train_model2 = model2.fit(X_train2, y_train2)


# In[139]:


#prediction for XGBoost model 1 and Final XGBoost Model 2
from sklearn.metrics import classification_report

pred1 = train_model1.predict(X_val2)
pred2 = train_model2.predict(X_val2)


# # Estimate Model Performance of XGBoost

# In[165]:


#Let's use accuracy score
from sklearn.metrics import accuracy_score, f1_score

print("Accuracy for XGBoost model 1: %.2f" % (accuracy_score(y_val2, pred1) * 100))
print("Accuracy for XGBoost model 2: %.2f" % (accuracy_score(y_val2, pred2) * 100))

print("F1 Score for XGBoost model 1: %.2f" % (f1_score(y_val2, pred1) * 100))
print("F1 Score for XGBoost model 2: %.2f" % (f1_score(y_val2, pred2) * 100))



# In[166]:


# CONFUSION MATRIX FOR FINAL XGBOOST MODEL 2
print('confusion matrix for Final XGBoost Model 2')
confusion_matrix(y_val2, pred2)


# In[135]:


# CONFUSION MATRIX FOR MODEL 1 - JUST TO COMPARE WITH XGBOOST MODEL 2 ABOVE
print('confusion matrix for XGBoost Model 1')
confusion_matrix(y_val2, pred1)


# # Model 3 - Random Forest

# In[94]:


# Random Forest
seed_value = 12321
model_rf1 = RandomForestClassifier(random_state=42,n_estimators = 10, min_samples_split= 4, min_samples_leaf=5, 
                                   max_depth=50, max_features=10)


# In[130]:


model_rf1.fit(X_train, y_train)


# # Estimate Model Performance of Random Forest 

# In[163]:


from sklearn.metrics import confusion_matrix

pred_val = model_rf1.predict(X_val)

print('confusion matrix for Final Random Forest Model')

confusion_matrix(y_val, pred_val)


# In[164]:


#Let's use accuracy score
from sklearn.metrics import accuracy_score, f1_score

print("Accuracy for model Final Random Forest Model: %.2f" % (accuracy_score(y_val, pred_val) * 100))
#print("Accuracy for model 2: %.2f" % (accuracy_score(y_val, pred2) * 100))

print("F1 Score for model Final Random Forest Model: %.2f" % (f1_score(y_val, pred_val) * 100))
#print("F1 Score for model 2: %.2f" % (f1_score(y_val, pred2) * 100))



# In[ ]:




