#!/usr/bin/env python
# coding: utf-8

# In[70]:


# [Rathini, Sivanandarajah]
# [Student number: 20220479]
# [MMA]
# [2021W]
# [MMA 869]
# [Due date: August 16,2020]


# Answer to Question [1], Part [1]


import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, silhouette_samples
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

import itertools

import scipy

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[72]:


#Read in Data, file:jewelry_customers.csv

#Before running change path to location where jewlry_customers.csv is saved

df = pd.read_csv("C:\\Users\srath\OneDrive\Documents\MMA COURSES\MMA 869 Machine Learning and AI\Assignment 1\jewelry_customers.csv")


# In[73]:


list(df)
df.info()
df.describe().transpose()
df.head(n=20)
df.tail()


# In[108]:


#Normalize the Data

X = df.copy()

X = X.drop(['ID'], axis=1)
X.head(10)
X.head(10)
scaler = StandardScaler()
#features = ['Age','Income','SpendingScore','Savings'] # all features except ID column
#X[features] = scaler.fit_transform(X[features])


scaler = StandardScaler()
X_scales = scaler.fit_transform(X)


# In[110]:


X.shape
X.info()
X.describe().transpose()
X.tail()


# In[165]:


# K-Means clustering 

k_means = KMeans(n_clusters=5, random_state=42)
k_means.fit(X_scales)


# In[166]:


k_means.labels_


# In[167]:


# Let's look at the centers
k_means.cluster_centers_


# In[168]:


#Internal Validation Metrics
# WCSS == Inertia
k_means.inertia_
print("K = 5")


# In[169]:


silhouette_score(X_scales, k_means.labels_)


# In[170]:


# In the case of K-Means, the cluster centers are the feature means - that's how K-Means is defined.
scaler.inverse_transform(k_means.cluster_centers_)


# In[171]:


# Let's look at some example rows in each.
for label in set(k_means.labels_):
    print('\nCluster {}:'.format(label))
    print(X[k_means.labels_==label].head())


# In[ ]:




