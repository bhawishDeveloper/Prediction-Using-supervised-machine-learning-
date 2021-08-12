#!/usr/bin/env python
# coding: utf-8

# In[30]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os


# In[31]:


os.getcwd()


# In[32]:


dataset = pd.read_csv('Iris.csv')
print(dataset)


# In[9]:


dataset.drop('Id', axis=1, inplace=True) 


# In[10]:


dataset.head()


# In[11]:


dataset.tail()


# In[21]:


dataset.describe()


# In[22]:


dataset.info()


# In[14]:


dataset.isnull().sum()


# In[23]:


dataset.corr()


# In[24]:


correlation = dataset.corr()
 
result = sns.clustermap(correlation, cmap ="YlGnBu");
plt.setp(result.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0)

result


# In[25]:


from sklearn.cluster import KMeans
x = dataset.iloc[:,0:4].values
WCSS=[]                      #Within-Cluster-Sum-of-Squares

for i in range(1,30):
    model = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    model.fit(x)
    WCSS.append(model.inertia_)  # kmeans.inertia_ returns the WCSS value for an initialized cluster
                                                                    
    
print(WCSS)


# In[29]:



plt.figure(figsize=(10, 6))
plt.plot(range(1, 30), WCSS, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('THE ELBOW METHOD')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()


# In[26]:


model= KMeans(n_clusters=3,random_state=0)
y= model.fit(x)
y=model.predict(x)
y


# In[28]:


from matplotlib import style
style.use('grayscale')
plt.figure(figsize=(10, 6))
plt.scatter(x[y == 0,2],x[y == 0,3], label= 'Iris-setosa', s=50)
plt.scatter(x[y == 1,2],x[y == 1,3], label= 'Iris-virginica',s=50)
plt.scatter(x[y == 2,2],x[y == 2,3], label= 'Iris-versicolor',s=50)

plt.scatter(model.cluster_centers_[:, 2], model.cluster_centers_[:, 3], s = 100, c = 'red', label = 'Centroids')
plt.legend()
plt.show()


# In[ ]:




