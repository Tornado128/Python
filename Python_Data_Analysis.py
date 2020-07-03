#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# data acquisition
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir("C:/Users/darbedar/Desktop/Python")
df = pd.read_csv("autos_imports-85.csv")


#df = pd.read_excel('autos_imports-85.xlsx')
#df = pd.read_cvs('autos_imports-85.txt',delimiter='\t')
#df = pd.read_csv("autos_imports-85.csv",header=None) #the header text becomes integre numbers
#df = pd.read_csv("autos_imports-85.csv", skiprows=1) #skip reading the first row 
#df = pd.read.csv("autos_imports-85.csv",header=None, names=['%symbolin1', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'wheel-base', 'length','width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower-peak-rpm', 'city-mpg', 'highway-mpg', 'price', 'Unnamed: 24', 'Unnamed: 25'])


# In[11]:


df.loc[:,'price'][0:5]


# In[12]:


df.iloc[1,5]


# In[13]:


df.columns


# In[28]:


df[['%symboling','normalized-losses']]
#df[df.columns]


# In[22]:


df.head()
df.iloc[0:4]


# In[24]:


#for index, row in df.iterrows():
#    print(index, row)


# In[ ]:


df.tail()


# In[38]:


df[df['normalized-losses']=='?']
#df.loc[df.columns=='?']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


df.to_csv("1.csv")


# In[ ]:


df.shape[0]


# In[ ]:


df.isna().sum().sum() 


# In[ ]:


df.dtypes


# In[ ]:


df.iloc[1,]


# In[ ]:


pd.set_option('max_columns', 100)
df.describe(include='all')


# In[ ]:


df["num-of-cylinders"].describe


# In[ ]:


df.dtypes


# In[ ]:


df.iloc[1,]


# In[ ]:


df["length"]


# In[ ]:


df.isna().sum()


# In[ ]:


df.isnull().sum()


# In[ ]:


df['make'].value_counts()


# In[ ]:


"?" in df.values
None in df.values
"Nan" in df.values
" " in df.values


# In[ ]:


df.iloc[:,1]


# In[ ]:


df.shape


# In[ ]:


pd.set_option('max_columns', 100)
df.describe(include='all')


# In[ ]:


# data acquisition
import pandas as pd
import numpy as np
import os
import re

os.chdir("C:/Users/darbedar/Desktop/Python")
df = pd.read_csv("autos_imports-85.csv")

df_m1=df.replace("?",np.NaN)
df_m1.isnull().sum()
df_m2=df_m1.dropna()
#df
#new_df1 = df[df["normalized-losses"] != "not available"]

#new_df2 = df[df["normalized-losses"]!= "not available"]
#new_df3 = df[df.loc[0:204, 1] != "not available"]
#print(new_df3)
df_m2


# In[ ]:


df_m2.shape


# In[ ]:


df_m2.iloc[:,0:2] 


# In[ ]:


df_m2.isnull().sum()


# In[ ]:


df_m2.dtypes
#df_m2.loc[:,'city-mpg']= df_m2.loc[:,'city-mpg'].astype(float)
#df_m2.loc[:,'city-mpg']=235/df_m2.loc[:,'city-mpg']

df_m2.loc[:,'city-mpg']= df_m2.loc[:,'city-mpg'].astype(float)
df_m2.loc[:,'city-mpg']=235/df_m2.loc[:,'city-mpg']


# In[ ]:


df_m3=df_m2.rename(columns={"city-mpg":"city-L/100 km"})


# In[ ]:


df_m3.dtypes


# In[ ]:


df_m3.loc[:,'city-L/100 km']


# In[ ]:


df_m3.loc[:,"length"]=(df_m3.loc[:,"length"]-df_m3.loc[:,"length"].mean())/df_m3.loc[:,"length"].std() #normalization


# In[ ]:


df_m3.loc[:,"length"].max()


# In[ ]:


#####Bining
df_m3.loc[:,"price"]=df_m3.loc[:,"price"].astype(float)


# In[ ]:


bins = np.linspace(min(df_m3.loc[:,"price"]),max(df_m3.loc[:,"price"]),4)
group_names=["Low","Medium","High"]
df_m3['price_binned']=pd.cut(df_m3.loc[:,"price"],bins,labels=group_names,include_lowest=True)
plt.hist(df_m3['price_binned'],bins=len(group_names))
plt.show()


# In[ ]:


plt.hist(df_m3.loc[:,"price"],bins=3,alpha=0.5,color='green')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




