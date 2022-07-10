#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd                                                 #pandas
import numpy as np                                                  #numpy
import sklearn                                                      # sklearn have datasets models and other things
import seaborn as sns                                               # seaborn 
import matplotlib.pyplot as plt
                    #to use datasets of sklearn
from sklearn.model_selection import train_test_split                # divide data in test and train
from sklearn.metrics import r2_score                                # residual error r square
from sklearn.metrics import mean_squared_error                      # mean square value
from sklearn import datasets, linear_model, metrics                 #import linera moduels
from sklearn.preprocessing import StandardScaler                    # scaler data for pca
from sklearn.decomposition import PCA                               #PCA 

from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge                              #ridge regression
from sklearn.linear_model import Lasso                              #lasso regression
from sklearn.neighbors import KNeighborsRegressor 
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor


# In[2]:


algo=[]
r2=[]
mserror=[]


# In[3]:


su=pd.read_csv('50_startup_regressor.csv')


# In[4]:


su.isnull().sum()


# In[5]:


su.State.unique()


# In[6]:


su.State=su.State.map({'New York':1, 'California':2, 'Florida':3})


# In[7]:


su


# In[8]:


(su==0).sum(axis=0) # check zero valuessu


# In[9]:


m=su["R&D Spend"].mean()
m


# In[10]:


s=su['Marketing Spend'].mean()
s


# In[11]:


su['R&D Spend']=su['R&D Spend'].replace(0,m)


# In[12]:


su['Marketing Spend']=su['Marketing Spend'].replace(0,s)


# In[13]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
vs = sc.fit_transform(su)
df = pd.DataFrame(vs)
X=df.iloc[:,[0,1,2,3]]
y=df.iloc[:,[4]]
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=1)


# In[28]:


y


# In[14]:


from sklearn.linear_model import LinearRegression 
alg='LRscaler'
lr=LinearRegression()
lr.fit(X_train, y_train)
y_pred=lr.predict(X_test)

score=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)


r2.append(score)
mserror.append(mse)
algo.append(alg)


# In[15]:




r2.append(score)
mserror.append(mse)
algo.append(alg)

from sklearn.neighbors import KNeighborsRegressor
alg='KNN'
knn = KNeighborsRegressor(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred2=knn.predict(X_test)
score=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)


r2.append(score)
mserror.append(mse)
algo.append(alg)

from sklearn import svm
alg='SVR'
sv= svm.SVR()
sv.fit(X_train, y_train)

y_pred=sv.predict(X_test)
score=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)


r2.append(score)
mserror.append(mse)
algo.append(alg)

from sklearn.tree import DecisionTreeRegressor
alg='DR'
dt= DecisionTreeRegressor()
dt.fit(X_train,y_train)
y_pred= dt.predict(X_test)
score=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)


r2.append(score)
mserror.append(mse)
algo.append(alg)


# In[16]:


from sklearn.preprocessing import MinMaxScaler
MM = MinMaxScaler()
vs = MM.fit_transform(su)
df = pd.DataFrame(vs)
X=df.iloc[:,[0,1,2,3]]
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=1)


# In[31]:


X


# In[17]:


from sklearn.linear_model import LinearRegression 
alg='LRmin max'
lr=LinearRegression()
lr.fit(X_train, y_train)
y_pred=lr.predict(X_test)

score=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)


r2.append(score)
mserror.append(mse)
algo.append(alg)


# In[18]:



from sklearn.neighbors import KNeighborsRegressor
alg='KNN2'
knn = KNeighborsRegressor(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred2=knn.predict(X_test)
score=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)


r2.append(score)
mserror.append(mse)
algo.append(alg)


# In[19]:



from sklearn import svm
alg='SVR2'
sv= svm.SVR()
sv.fit(X_train, y_train)

y_pred=sv.predict(X_test)
score=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)


r2.append(score)
mserror.append(mse)
algo.append(alg)


# In[20]:



from sklearn.tree import DecisionTreeRegressor
alg='DR2'
dt= DecisionTreeRegressor()
dt.fit(X_train,y_train)
y_pred= dt.predict(X_test)
score=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)


r2.append(score)
mserror.append(mse)
algo.append(alg)


# In[21]:


plt.plot(algo,r2)


# In[22]:


plt.plot(algo,mserror)


# In[23]:


su


# In[24]:




