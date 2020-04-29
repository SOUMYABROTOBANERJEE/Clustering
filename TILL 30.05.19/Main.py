#!/usr/bin/env python
# coding: utf-8

# In[99]:


import pandas as pd


# In[100]:


import numpy as np


# In[101]:


from matplotlib import pyplot as plt


# In[ ]:





# In[102]:


from sklearn.datasets import load_boston


# In[103]:


boston=load_boston()
x=boston.data
y=boston.target
print(boston.keys())


# In[104]:


columns=boston.feature_names
columns


# In[105]:


print(boston.DESCR)


# In[106]:


boston_df=pd.DataFrame(boston.data)


# In[107]:


boston_df.columns=columns
boston_df_o=boston_df
boston_df.shape


# In[108]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(x=boston_df['DIS'])


# In[109]:


fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(boston_df['INDUS'], boston_df['TAX'])
ax.set_xlabel('Proportion of non-retail business acres per town')
ax.set_ylabel('Full-value property-tax rate per $10,000')
plt.show()


# In[110]:


from scipy import stats


# In[111]:


z = np.abs(stats.zscore(boston_df)) 


# In[ ]:





# In[112]:


print(z)


# In[113]:


print(np.where(z>3))


# In[114]:


boston_df_o =  boston_df[(z < 3).all(axis=1)]


# In[115]:


boston_df.shape
boston_df_o.shape


# In[93]:


boston_df.shape


# In[94]:


fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(boston_df['B'], boston_df['TAX'])
plt.ylim(100, 800)
plt.xlim(-100, 500)
plt.title("Original")
plt.show()


# In[87]:


fig = plt.subplots(figsize=(16,8))
plt.scatter(boston_df_o['B'], boston_df_o['TAX'])
plt.ylim(100, 800)
plt.xlim(-100, 500)
plt.xlabel("B")
plt.ylabel("TAX using z score")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[88]:


boston_df_o.shape


# In[89]:


boston_df_o1 = boston_df


# In[90]:


Q1 = boston_df_o1.quantile(0.25)
Q3 = boston_df_o1.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

boston_df_out = boston_df_o1[~((boston_df_o1 < (Q1 - 1.5 * IQR)) |(boston_df_o1 > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[51]:


fig= plt.subplots(figsize=(16,8))
plt.scatter(boston_df_out['B'], boston_df_out['TAX'])
plt.ylim(100, 800)
plt.xlim(-100, 500)
plt.xlabel("B")
plt.ylabel("TAX using IQR")
plt.show()
boston_df_out.shape


# In[ ]:




