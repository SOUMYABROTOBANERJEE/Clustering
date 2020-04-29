#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_boston
from scipy import stats

boston=load_boston()
x=boston.data
y=boston.target
print(boston.keys())
df=pd.DataFrame(boston.data)
columns=boston.feature_names
df.columns=columns
cluster_label=[]

np.random.seed(20)
k = 3
# centroids[i] = [x, y]
centroids = {
    i+1: [np.random.randint(0,20), np.random.randint(0,500)]
    for i in range(k)
}
    
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


# In[29]:


def assignment(df, centroids):
    k=0
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['INDUS'] - centroids[i][0]) ** 2
                + (df['TAX'] - centroids[i][1]) ** 2
            )
        )
      
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    #print(centroid_distance_cols)
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
            if(z>400):
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


# In[30]:


import copy

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


# In[31]:


centroids


# In[32]:


old_centroids


# In[33]:


df = assignment(df, centroids)

# Plot results
fig = plt.figure(figsize=(16, 8))
plt.scatter(df['INDUS'], df['TAX'],color=df['color'], alpha=0.5, edgecolor='k')
plt.ylim(0,800)
plt.xlim(0,35)
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.show()


# In[34]:


while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df['closest']): #that is no change in the centroids
        break
        

        

#OUTLIER REMOVAL
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
#print(IQR)
#df_o =df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]




fig = plt.figure(figsize=(16, 8))
plt.scatter(df['INDUS'], df['TAX'],color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.show()


# In[35]:


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
plt.scatter(df['distance_from_1'],df['Probability_1'])


# In[36]:


df


# In[38]:


df.to_csv('kmeans.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




