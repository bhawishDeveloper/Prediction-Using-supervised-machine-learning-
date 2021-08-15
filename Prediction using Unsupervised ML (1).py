#!/usr/bin/env python
# coding: utf-8

# In[92]:


import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as bplt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.cluster import KMeans


# In[46]:


bhdata = pd.read_csv('Iris.csv')
print(bhdata)


# In[47]:


bhdata.drop('Id', axis=1, inplace=True) 
bhdata


# In[48]:


bhdata.head()


# In[49]:


bhdata.tail()


# In[21]:


dataset.describe()


# In[50]:


bhdata.shape


# In[51]:


bhdata.info()


# In[52]:


bhdata.isnull().sum()


# In[73]:


x = bhdata.iloc[:,0:4].values
wcss=[] 

for i in range(1,15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)  
                                                                    
    
print(wcss)


# In[83]:


bplt.plot(range(1, 15), wcss,'r', color='green')
bplt.plot(wcss, 'ro')
bplt.title('The Elbow Method')
bplt.xlabel('Number of clusters')
bplt.ylabel('Within Clusters Sum of Squares')
bplt.grid()
bplt.show()


# In[89]:


model= KMeans(n_clusters=3, random_state=0, init = 'k-means++')
bhpredict = model.fit_predict(x)


# In[91]:


bplt.scatter(x[bhpredict == 0,0],x[bhpredict == 0,1], label= 'Iris-setosa', s=50, c='green')
bplt.scatter(x[bhpredict == 1,0],x[bhpredict == 1,1], label= 'Iris-virginica', s=50, c = 'red')
bplt.scatter(x[bhpredict == 2,0],x[bhpredict == 2,1], label= 'Iris-versicolor', s=50, c = 'blue')

bplt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
bplt.legend()
bplt.grid()
bplt.show()


# In[ ]:




