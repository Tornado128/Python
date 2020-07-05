#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures # importing libraries for polynomial transform
from sklearn.pipeline import Pipeline                # for creating pipeline
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


# In[2]:


os.chdir("C:/Users/darbedar/Desktop/Python")
df = pd.read_csv("autos_imports-85.csv")


# In[3]:


df.columns


# In[4]:


df.dtypes


# In[5]:


df=df.replace('?',np.NaN)
df=df.dropna()
df.head(5)
df['city-mpg'].value_counts()
df['price'].value_counts()


# In[6]:


df.loc[:,'city-mpg']=df.loc[:,'city-mpg'].astype(float)
df.loc[:,'price']=df.loc[:,'price'].astype(float)


# In[7]:


lm=LinearRegression()
x=df[['city-mpg']]
y=df.loc[:,'price']
lm.fit(x,y)
[lm.intercept_, lm.coef_]
lm.score(x, y) #R^2
plt.scatter(x,y,color='g')
plt.plot(x,lm.predict(x),color='k')


# In[8]:


df.head(5)


# In[9]:


x2=df[['length','city-mpg']]
y2=df['price']
lm.fit(x2,y2)
yhat=lm.predict(x2)
lm.score(x2,y2) #R2
[lm.intercept_, lm.coef_]


# In[10]:


#plt.scatter(x2,y2,color='g')
#plt.plot(x2,yhat,color='k')
#### how to visualize


# In[11]:


ax1 = sns.distplot(df['price'],hist=False, color='r', label="actual values")
sns.distplot(yhat, hist=False, color='b', label="fitted values", ax=ax1)


# In[12]:


#####polynomial fit
x=df['city-mpg']
y=df.loc[:,'price']
f = np.polyfit(x,y,2) #order of polynomial
p=np.poly1d(f)
r2_score(y, p(x))
plt.scatter(x,y,color='g')
plt.plot(x,p(x),'.',color='k')


# In[19]:


pr=PolynomialFeatures(degree=2,include_bias=False)
pr.fit_transform(df[['length','city-mpg']])                # make x1, x2, x1x2, (x1)^2, (x2)^2
SCALE=MinMaxScaler()
SCALE.fit(df[['length','city-mpg']])
x_scale=SCALE.transform(df[['length','city-mpg']])         #scale based on min and max (0,1)
Input=[('scale',MinMaxScaler()),('polynomial',PolynomialFeatures(degree=2)),('modal',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(df[['length','city-mpg']],df.loc[:,'price'])
yhat1=pipe.predict(df[['length','city-mpg']])
ax1 = sns.distplot(df['price'],hist=False, color='r', label="actual values")
sns.distplot(yhat1, hist=False, color='b', label="fitted values", ax=ax1)
r2 = r2_score(df['price'],yhat1)
print(r2)


# In[25]:


pr=PolynomialFeatures(degree=2,include_bias=False)
pr.fit_transform(df[['length','city-mpg']])                # make x1, x2, x1x2, (x1)^2, (x2)^2


# In[14]:


SCALE


# In[15]:


####Polynomial Regreesion
#### Y = b0 + b1x1 + b2x2 + b3x1x2 + b4(x1)^2+b5(x2)^5+
df['length_norm']=(df['length']-df['length'].min())/(df['length'].max()-df['length'].min())
df['city-mpg_norm']=(df['city-mpg']-df['city-mpg'].min())/(df['city-mpg'].max()-df['city-mpg'].min())
df['(length_norm)^2']=df['length_norm']*df['length_norm']
df['(city-mpg_norm)^2']=df['city-mpg_norm']*df['city-mpg_norm']
df['(length_norm).(city-mpg_norm)']=df['length_norm']*df['city-mpg_norm']
x = df[['length_norm','city-mpg_norm','(length_norm).(city-mpg_norm)','(length_norm)^2','(city-mpg_norm)^2']]
y = df['price']
lm.fit(x,y)
yhat2 = lm.predict(x)
print('R^2 is equal to',lm.score(x,y))
print([lm.intercept_,lm.coef_])
ax1 = sns.distplot(df['price'],hist=False, color='r', label="actual values")
sns.distplot(yhat2, hist=False, color='b', label="fitted values", ax=ax1)


# In[16]:


pr=PolynomialFeatures(degree=2,include_bias=False)
pr.fit_transform(df[['length','city-mpg']])                # make x1, x2, x1x2, (x1)^2, (x2)^2
SCALE=MinMaxScaler()
SCALE.fit(df[['length','city-mpg']])
x_scale=SCALE.transform(df[['length','city-mpg']])


# In[17]:


Input


# In[ ]:




