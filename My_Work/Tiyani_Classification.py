#!/usr/bin/env python
# coding: utf-8

# In[279]:


import string 
import nltk
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import f1_score


#from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler


# In[243]:


#Loading data
test = pd.read_csv("D:/Data Science/Classification/Data/test.csv")
train = pd.read_csv("D:/Data Science/Classification/Data/train.csv")


# In[244]:


test.head()


# In[245]:


train.head()


# In[284]:


#from sklearn.utils import resample
#believe = train[train["sentiment"]==1]
#no_believe = train[train["sentiment"]==-1]
#neutral = train[train["sentiment"]==0]
#news = train[train["sentiment"]==2]

#Upsample minority



#no_believe_upsampled = resample(no_believe,replace=True,n_sample = a,random_state = 27 )
#neutral_upsampled = resample(neutral,replace = True, random_state = 27)
#news_upsampled = resample(news,replace = True, random_state = 27)
#upsampled = pd.concat([believe,no_believe_upsampled,neutral_upsampled,news_upsampled])


#df_class_1_over = df_class_1.sample(count_class_0, replace=True)
#df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

#no_believe_upsampled = no_believe.sample(believe, replace=True)
#neutral_upsampled = neutral.sample(believe,replace=True)
#news_upsampled = news.sample(believe, replace=True)
#upsampled = pd.concat([believe,no_believe_upsampled,neutral_upsampled,news_upsampled], axis =0)


#splitting our data
y = train["sentiment"]
X = train["message"]

ros = RandomOverSampler()
X_ros, y_ros = ros.fit_sample(X, y)


# In[302]:


vectorizer = TfidfVectorizer(ngram_range= (1,2), min_df= 2, stop_words= "english")
X_vectorized = vectorizer.fit_transform(X)


# In[433]:


X_train,X_val,y_train,y_val = train_test_split(X_vectorized,y,test_size= 0.3,shuffle = True, stratify = y, random_state = 61)


# In[434]:


rfc = XGBClassifier()

#rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_val)


# In[435]:


f1_score(y_val, rfc_pred, average = "macro")


# In[436]:


testx = test["message"]
test_vect = vectorizer.transform(testx)


# In[437]:


y_pred = rfc.predict(test_vect)


# In[438]:


test["sentiment"] = y_pred


# In[439]:


test.head()


# In[440]:


test[["tweetid","sentiment"]].to_csv("sub.csv", index = False)


# In[ ]:





# In[ ]:




