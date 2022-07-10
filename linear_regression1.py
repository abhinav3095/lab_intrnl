


import pandas as pd
import numpy as np
#%matplotlib inline
import sklearn
import seaborn as sn
from sklearn import datasets, linear_model, metrics





boston = datasets.load_boston(return_X_y=False)





X = boston.data




X.shape




y= boston.target
y.shape





boston.keys()


# In[6]:


df = pd.DataFrame(boston['data'],columns=boston['feature_names'])


# In[7]:


df.head()


# In[17]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=12)


# In[18]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[19]:


reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)


# In[20]:


y_pred = reg.predict(X_test)
y_pred


# In[21]:


y_test


# In[22]:


from sklearn.metrics import r2_score
score= r2_score(y_test,y_pred)
score


# In[23]:


print('Coefficients: \n',reg.coef_)


# In[24]:


import matplotlib.pyplot as plt


# In[25]:


plt.style.use('fivethirtyeight')
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
plt.hlines(y=0,xmin=0,xmax=50,linewidth=2)


# In[49]:


from sklearn.model_selection import cross_val_score,KFold, StratifiedKFold
cv = KFold(n_splits=6)
score = cross_val_score(reg,X_train, y_train,cv=cv,scoring='r2')
print(score)
score.mean()

