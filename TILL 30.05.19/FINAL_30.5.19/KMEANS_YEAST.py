#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
df=pd.read_csv('yeast_csv.csv')
df.drop(columns='class_protein_localization',inplace=True)
columns=df.columns

np.random.seed(0)
k = 3
# centroids[i] = [x, y]
centroids=[]
for i in range(3):
    centroids.append(np.random.uniform(size=8))


# In[2]:


len(columns)


# In[3]:


for i in range(k):
    print(centroids[i])


# In[4]:


def assignment(df, centroids):
    k=3
    for i in range(k):
      
        for j in range(len(columns)):
            
            x=np.sqrt((df[columns[j]] - centroids[i][j]) ** 2)
            
                    
            
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
            df['distance_from_{}'.format(i)] =x
            
      
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in range(k)]
    #print(centroid_distance_cols)
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    #cluster_label.append(df.loc[:, centroid_distance_cols].idxmin(axis=1))
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    #df['color'] = df['closest'].map(lambda x: colmap[x])  
    return df


# In[5]:


assignment(df,centroids)


# In[6]:


import copy

old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in range(3):
        centroids[i][0] = np.mean(df[df['closest'] == i]['mcg'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['gvh'])
        centroids[i][2] = np.mean(df[df['closest'] == i]['alm'])
        centroids[i][3] = np.mean(df[df['closest'] == i]['mit'])
        centroids[i][4] = np.mean(df[df['closest'] == i]['erl'])
        centroids[i][5] = np.mean(df[df['closest'] == i]['pox'])
        centroids[i][6] = np.mean(df[df['closest'] == i]['vac'])
        centroids[i][7] = np.mean(df[df['closest'] == i]['nuc'])
        #centroids[i][8] = np.mean(df[df['closest'] == i]['TAX'])
    return k

centroids = update(centroids)


# In[7]:


df = assignment(df, centroids)


# In[8]:


while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df['closest']): #that is no change in the centroids
        break


# In[9]:


def prob(df,centroids):
    for i in range(3):
            # sqrt((x1 - x2)^2 - (y1 - y2)^2)
            df['Probability_{}'.format(i)] =(np.exp(-1*
                (
            np.sqrt(
                (df['mcg'] - centroids[i][0]) ** 2
                + (df['gvh'] - centroids[i][1]) ** 2
                + (df['alm'] - centroids[i][2]) ** 2
                + (df['mit'] - centroids[i][3]) ** 2
                + (df['erl'] - centroids[i][4]) ** 2
                + (df['pox'] - centroids[i][5]) ** 2
                + (df['vac'] - centroids[i][6]) ** 2
                + (df['nuc'] - centroids[i][7]) ** 2
            )
        )
            )/sum(df,centroids)
            )

            centroid_distance_cols1 = ['Probability_'.format(i) for i in range(3)]
            
def sum(df,centroids): 
        sums=0
        for i in range(3):
            
            sums=sums+(np.exp(-1*
                    np.sqrt(
                        (
            np.sqrt(
                (df['mcg'] - centroids[i][0]) ** 2
                + (df['gvh'] - centroids[i][1]) ** 2
                + (df['alm'] - centroids[i][2]) ** 2
                + (df['mit'] - centroids[i][3]) ** 2
                + (df['erl'] - centroids[i][4]) ** 2
                + (df['pox'] - centroids[i][5]) ** 2
                + (df['vac'] - centroids[i][6]) ** 2
                + (df['nuc'] - centroids[i][7]) ** 2
            )
        )
                    )
                )
                )
        return (sums)


prob(df,centroids)
print(df.head())


# In[10]:


df.to_csv('KMEANS_YEAST.csv')


# In[ ]:





# In[ ]:





# In[14]:





# In[16]:





# In[17]:





# In[18]:





# In[19]:





# In[25]:





# In[ ]:





# In[ ]:





# In[ ]:




