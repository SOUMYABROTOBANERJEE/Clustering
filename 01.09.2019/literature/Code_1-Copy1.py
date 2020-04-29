#!/usr/bin/env python
# coding: utf-8

# In[64]:


from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.mixture import GaussianMixture
import warnings
import glob
import cv2
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import skfuzzy as fuzz
import os


# In[65]:


os.listdir("C:\\Users\\soumy\\Desktop\\JU\\01.09.2019\\literature\\literature\\")


# In[2]:


k=2
thresh=0.2
limit=20


# In[3]:


DataSet="ALOI"
path="C:\\Users\\soumy\\Desktop\\JU\\01.09.2019\\literature\\literature\\"+DataSet+"\\"+DataSet+"_withoutdupl_norm.arff"
#path_img= "C:\\Users\\soumy\\Desktop\\JU\Data for Outlier\\Data for Outlier\\001\\"


# In[ ]:





# In[4]:


data = arff.loadarff(path)


# In[5]:


df12=pd.DataFrame(data[0])
#df=df[df['class']=='C001']
from copy import deepcopy
df_copy=deepcopy(df12)


# In[ ]:





# In[6]:


labelsss=df12['outlier']


# In[7]:


#for i in range(len(labelsss)):
#   if(labelsss[i][0]==' '):
#        labelsss[i]=labelsss[i][7:11]
#    else:
#        labelsss[i]=labelsss[i][6:9]
    


# In[8]:


np.unique(labelsss,return_counts=True)


# In[9]:


df12.drop(columns=['outlier','id'],inplace=True)


# In[10]:


dictfinal={}


# In[11]:


dict1={}
dictoutlier={}
dictnotoutlier={}


# In[12]:


from sklearn import preprocessing

x = df12.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df12 = pd.DataFrame(x_scaled)


# In[13]:


df12.head()


# In[14]:


pd.set_option('use_inf_as_na', True)
columns=df12.columns
df=df12
silhouette=[]
for i in range(limit):
    df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    df=df.reset_index()
    df.drop(columns='index',inplace=True)
    gmmkpp = GaussianMixture(n_components=k,random_state=100).fit(df)
    gmmkm = GaussianMixture(init_params='random',n_components=k,random_state=100).fit(df)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(df.T, k, 2, error=0.005, maxiter=1000, init=None)
    
    labelskmpp = gmmkpp.predict_proba(df)
    s1=silhouette_score(df, np.argmax(labelskmpp,axis=1))
    labelskm = gmmkm.predict_proba(df)
    s2=silhouette_score(df, np.argmax(labelskm,axis=1))
    s3=silhouette_score(df, np.argmax(u,axis=0))
    s1=abs(s1)
    s2=abs(s2)
    s3=abs(s3)
    
    y_net = s1*labelskmpp + s2*labelskm + s3*u.T
    
    #print(y_net)
    ################## SCORE ###############
    silhouette_avg = silhouette_score(df, np.argmax(y_net,axis=1))
    print('Silhouette Score =',silhouette_avg,'for i =',i+1)
    silhouette.append(silhouette_avg)
    ############### OUTLIER ########################
    if(i==0):
        outliers=np.where(np.max(y_net,axis=1) < thresh)
    else :
        thresh=thresh
        outliers=np.where(np.max(y_net,axis=1) < thresh)
    #if(len(outliers[0])==0):
        #print('All Outliers Removed.... Breaking')
        #break
    print(outliers[0])
    dict1.update({i:outliers[0]})
    mol=np.delete(y_net,outliers[0],axis=0)
    y_net=mol
    df=df.drop(outliers[0])
    df=df.reset_index()
    df.drop(columns='index',inplace=True)
    ylabel=np.argmax(y_net,axis=1)


    labels=pd.Series(ylabel,name="Labels")




    print(np.unique(labels,return_counts=True))
    df=df.join(labels)

    df.update(labels)
    dictoutlier.update({i:list(df[df['Labels']==1].index)})
    dictnotoutlier.update({i:list(df[df['Labels']==0].index)})

    centroids=df.groupby('Labels', as_index=False)[columns].mean()
    centroids=centroids.drop(columns='Labels')
    centroids=centroids.values.tolist()

    for i1 in range(k):
            x=0
            y=0
            for j1 in range(len(columns)):
                x+=np.sqrt(np.abs((df[columns[j1]] - centroids[i1][j1]) ** 2))
            y=np.exp(-1*x)   
            # sqrt((x1 - x2)^2 - (y1 - y2)^2)
            df['distance_from_{}'.format(i1)] =x
            df['Probability_of_{}'.format(i1)]=y



    for j2 in range(k):
        df1=df[df['Labels']==j2]
        for i2 in columns:
            df1[i2]=np.abs(df[i2]-df1['Probability_of_{}'.format(j2)]/df1['distance_from_{}'.format(j2)])
            #print(df1.head())
            df.update(df1[i2])

    df=df[columns]

    #print(df.head(2))
    print('----------------------------------------------------------------------------------------')
    print()


    
    
    
    
    
    


# In[24]:





# In[25]:



m=np.argmax(silhouette)

print('Best for m='+str(m+1)+' >=> '+ str(max(silhouette)))


# In[31]:





# In[ ]:





# In[59]:


z=[]
y1=[]
for i in dictoutlier[m]:
    z.append(labelsss[i])
    
f00=np.unique(z,return_counts=True)
z=str(list(foo[0])+list(foo[1]))


# In[60]:



for i in dictnotoutlier[m]:
    y1.append(labelsss[i])
    
foo=np.unique(y1,return_counts=True)
y1=str(list(foo[0])+list(foo[1]))


# In[ ]:





# In[67]:


fig=plt.figure()
f=plt.plot(silhouette)
plt.title(DataSet+' for  k = '+ str(k) )
plt.xlabel("Iteration -----> \n MAX for "+str(m+1)+"\n OUTLIER:"+z+"\n NON-OUTLIER:"+y1)
plt.ylabel("Average Silhouette Score")

plt.savefig('C:\\Users\\soumy\\Desktop\\JU\\01.09.2019\\literature\\Results\\'+DataSet+'.png',bbox_inches='tight')


# In[20]:


#from sklearn.ensemble import IsolationForest

#clf = IsolationForest(n_estimators=100, warm_start=True)
#y=clf.fit_predict(df12)  # fit 10 trees  


# In[ ]:





# In[21]:


#np.unique(y,return_counts=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


#from sklearn_extensions.fuzzy_kmeans import KMedians, FuzzyKMeans, KMeans


# In[23]:


#kmeans = KMeans(k=2)
#kmeans.fit(df12)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




