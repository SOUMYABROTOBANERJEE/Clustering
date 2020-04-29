#!/usr/bin/env python
# coding: utf-8

# In[262]:


import pandas as pd
import numpy as np


# In[263]:


import matplotlib.pyplot as plt
colmap = {1: 'r', 2: 'g', 3: 'b'}


# In[264]:


df1=pd.read_csv('kmeans.csv')


# In[265]:


df1.shape


# In[266]:


df2=pd.read_csv('kmeanspp.csv')


# In[267]:


df2.shape


# In[268]:


df1.head()


# In[269]:


sums=[]
def sum(df):
    sums.append(df1['Probability_1']+df2['Probability_1']+df1['Probability_2']+df2['Probability_2']+df1['Probability_3']+df2['Probability_3'])
    return sums
sum(df)
sums


# In[270]:


df1['Probability1']=(df1['Probability_1']+df2['Probability_1'])/sums[0]
df1['Probability2']=(df1['Probability_2']+df2['Probability_2'])/sums[0]
df1['Probability3']=(df1['Probability_3']+df2['Probability_3'])/sums[0]


# In[271]:


def assignment(df):
    
    counter_k = ['Probability{}'.format(i+1) for i in range(3)]
    #print(centroid_distance_cols)
    df['cluster_label'] = df.loc[:, counter_k].idxmax(axis=1)
    #df.loc[:, centroid_distance_cols].max(axis=1)
    df['Max_prob']=(df.loc[:, counter_k].max(axis=1))
    df['Min_prob']=(df.loc[:, counter_k].min(axis=1))
    df['cluster_label'] = df['cluster_label'].map(lambda x: int(x.lstrip('Probability')))
    df['color'] = df['cluster_label'].map(lambda x: colmap[x]) 
    return df

assignment(df1)


# In[272]:


def outliers(df):
    for i in range(3):
        for index,row in df.iterrows():
            z=6.610209e-170
            if(row['Min_prob']<z):
                df.drop(index,inplace=True)
                print(index)
    return df


# In[273]:


outliers(df1)
df1.shape


# In[274]:


def final_cluster_plot(df):
    fig=plt.subplots(figsize=(16,8))
    plt.scatter(df['INDUS'], df['TAX'],color=df['color'], alpha=0.5, edgecolor='k')
    plt.ylim(0,800)
    plt.xlim(0,35)
    plt.show()
    
final_cluster_plot(df1)


# In[275]:


df1.to_csv('Final_Output.csv')    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[276]:


#MY CUSTOM CLASSIFIER SCORE


# In[277]:



def score_metric(df):
    y=np.array(df[['INDUS', 'TAX']])
    silhouette_avg = silhouette_score(y, np.array(df['cluster_label']))
    print(silhouette_avg)#(-1,1) Higher the better
    
score_metric(df1)


# In[ ]:





# In[278]:


#AUTO SCIKIT LEARN KPP CLUSTERING ALGORITHM


# In[279]:


from sklearn.datasets import load_boston
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
y=np.array(df[['INDUS', 'TAX']])
clusterer = KMeans(n_clusters=3, random_state=10)
cluster_labels = clusterer.fit_predict(y)
silhouette_avg = silhouette_score(y,cluster_labels)
print(silhouette_avg)#(-1,1) Higher the better


# In[ ]:




