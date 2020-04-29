#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np


# In[38]:


k=3


# In[39]:


df1=pd.read_csv('KMEANS_YEAST.csv')


# In[40]:


df1.shape


# In[41]:


df2=pd.read_csv('KMEANS_PP_YEAST.csv')


# In[42]:


df2.shape


# In[43]:


df1.head()


# In[44]:


sums=[]
def sum():
    sums.append(df1['Probability_0']+df2['Probability_0']+df1['Probability_1']+df2['Probability_1']+df1['Probability_2']+df2['Probability_2'])
    return sums
sum()
sums


# In[45]:


df1['Probability0']=(df1['Probability_0']+df2['Probability_0'])/sums[0]
df1['Probability1']=(df1['Probability_1']+df2['Probability_1'])/sums[0]
df1['Probability2']=(df1['Probability_2']+df2['Probability_2'])/sums[0]


# In[46]:


def assignment(df):
    
    counter_k = ['Probability{}'.format(i) for i in range(3)]
    #print(centroid_distance_cols)
    df['cluster_label'] = df.loc[:, counter_k].idxmax(axis=1)
    #df.loc[:, centroid_distance_cols].max(axis=1)
    df['Max_prob']=(df.loc[:, counter_k].max(axis=1))
    df['Min_prob']=(df.loc[:, counter_k].min(axis=1))
    df['cluster_label'] = df['cluster_label'].map(lambda x: int(x.lstrip('Probability')))
    #df['color'] = df['cluster_label'].map(lambda x: colmap[x]) 
    return df

assignment(df1)


# In[49]:


def outliers(df):
    for i in range(3):
        for index,row in df.iterrows():
            z=0.25
            if(row['Min_prob']<z):
                df.drop(index,inplace=True)
                print(index)
    return df


# In[50]:


outliers(df1)
df1.shape


# In[12]:


#def final_cluster_plot(df):
    #fig=plt.subplots(figsize=(16,8))
    #plt.scatter(df['INDUS'], df['TAX'],color=df['color'], alpha=0.5, edgecolor='k')
    #plt.ylim(0,800)
    #plt.xlim(0,35)
    #plt.show()
    
#final_cluster_plot(df1)


# In[51]:


df1.to_csv('Final_Output_Yeast.csv')    


# In[ ]:





# In[52]:


df1.shape


# In[ ]:





# In[ ]:





# In[26]:


#MY CUSTOM CLASSIFIER SCORE


# In[53]:


from sklearn.metrics import silhouette_samples, silhouette_score
def score_metric(df):
   
    silhouette_avg = silhouette_score(df1, np.array(df['cluster_label']))
    print(silhouette_avg)#(-1,1) Higher the better
    
score_metric(df1)


# In[ ]:





# In[278]:


#AUTO SCIKIT LEARN KPP CLUSTERING ALGORITHM


# In[28]:


from sklearn.datasets import load_boston
from sklearn.cluster import KMeans

clusterer = KMeans(n_clusters=3, random_state=10)
cluster_labels = clusterer.fit_predict(df1)
silhouette_avg = silhouette_score(df1,cluster_labels)
print(silhouette_avg)#(-1,1) Higher the better


# In[ ]:




