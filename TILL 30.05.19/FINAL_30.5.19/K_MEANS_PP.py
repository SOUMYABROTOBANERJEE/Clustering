#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_boston
from scipy import stats
import copy

boston=load_boston()
X=boston.data
y=boston.feature_names
print(boston.keys())
df=pd.DataFrame(X)
df.columns=y
cluster_label=[]
#centroids=dict(centro)


# In[56]:


np.random.seed(20)
k = 3
# centroids[i] = [x, y]
centro=[[np.random.randint(0,20), np.random.randint(0,500)],[np.random.randint(0,20), np.random.randint(0,500)],[np.random.randint(0,20), np.random.randint(0,500)]]

def centroid(df):
    for i in range(k):
        if i!=0:
            sum = np.linalg.norm(df[['INDUS', 'TAX']].sub(np.array(centro[i-1])), axis=1)
            index=int(np.argmax(sum))
            x=df['INDUS'][index]
            y=df['TAX'][index]
            centro[i]=[x ,y]
centroid(df)
df.head()
centroids={}
#converting into dict
for i in range(k):
    centroids[i+1]=centro[i]


# In[57]:


fig=plt.subplots(figsize=(16,8))
plt.scatter(df['INDUS'],df['TAX'])
plt.ylim(0,800)
plt.xlim(0,35)
colmap = {1: 'r', 2: 'g', 3: 'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.show()

for i in centroids.keys():
    print(centroids[i])


# In[58]:


def assignment(df, centroids):
    k=0
    cluster_label=[]
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['INDUS'] - centroids[i][0]) ** 2
                + (df['TAX'] - centroids[i][1]) ** 2
            )
        )
    
      
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    cluster_label.append(df.loc[:, centroid_distance_cols].idxmin(axis=1))
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])  
    return df


    
def outliers(df,centroid):
    for i in centroid.keys():
        for index,row in df.iterrows():
            z=np.sqrt((row['INDUS']-centroid[i][0])**2+(row['TAX']-centroid[i][1])**2)
            #print(z)
            if(z>500):
                df.drop(index,inplace=True)
                #print(index)
    return df

df = assignment(df, centroids)
print(df.head())
print(df.shape)

fig = plt.figure(figsize=(16, 8))
plt.scatter(df['INDUS'], df['TAX'],color=df['color'], alpha=0.5, edgecolor='k')
plt.ylim(0,800)
plt.xlim(0,35)
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.show()


# In[59]:


old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['INDUS'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['TAX'])
    return k

centroids = update(centroids)
    
fig = plt.figure(figsize=(16, 8))
plt.scatter(df['INDUS'], df['TAX'],color=df['color'], alpha=0.5, edgecolor='k' )
plt.ylim(0,800)
plt.xlim(0,35)
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.show()


# In[60]:


print(centroids)
print('---------------------------------')
print(old_centroids)


# In[61]:


df = assignment(df, centroids)

# Plot results
fig = plt.figure(figsize=(16, 8))
plt.scatter(df['INDUS'], df['TAX'],color=df['color'], alpha=0.5, edgecolor='k')
plt.ylim(0,800)
plt.xlim(0,35)
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.show()


# In[62]:


while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df['closest']): #that is no change in the centroids
        break
        


# In[63]:


#final results

fig = plt.figure(figsize=(16, 8))
plt.scatter(df['INDUS'], df['TAX'],color=df['color'], alpha=0.5, edgecolor='k')
plt.ylim(0,800)
plt.xlim(0,35)
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.show()


# In[64]:


def prob(df,centroids):
    for i in centroids.keys():
            # sqrt((x1 - x2)^2 - (y1 - y2)^2)
            df['Probability_{}'.format(i)] =(np.exp(-1*
                np.sqrt(
                    (df['INDUS'] - centroids[i][0]) ** 2
                    + (df['TAX'] - centroids[i][1]) ** 2
                )
            )/sum(df,centroids)
            )

            centroid_distance_cols1 = ['Probability_'.format(i) for i in centroids.keys()]
            
def sum(df,centroids): 
        sum=0
        for i in centroids.keys():
            
            sum=sum+(np.exp(-1*
                    np.sqrt(
                        (df['INDUS'] - centroids[i][0]) ** 2
                        + (df['TAX'] - centroids[i][1]) ** 2
                    )
                )
                )
        return (sum)


prob(df,centroids)
print(df.head())

fig=plt.figure(figsize=(16,8))
plt.scatter(df['Probability_2'],df['distance_from_2'])


# In[65]:


df.head()


# In[66]:


df.to_csv('kmeanspp.csv')


# In[67]:


df.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




