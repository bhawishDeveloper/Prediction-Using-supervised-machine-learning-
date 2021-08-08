#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as mplt
from sklearn.linear_model import LinearRegression


# In[9]:


data = pd.read_csv('http://bit.ly/w-data')
print(data)

print('Data Import: Successful')


# In[10]:


data.head()


# In[11]:


data.tail()


# In[12]:


data.info()


# In[13]:


data.describe()


# In[21]:


mplt.figure(figsize=(8,4))
mplt.scatter(data['Hours'],data['Scores'], color='blue', marker='*')
mplt.xlabel('Study Hours')
mplt.ylabel('Score Count')
mplt.title('Comparison of Study hours with Score count')
mplt.grid()


# In[32]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data.Hours, data.Scores, test_size=0.2, random_state = 42)


# In[28]:


mplt.scatter(x_train, y_train, label = 'Training Data', color='b')
mplt.scatter(x_test, y_test, label = 'Testing Data', color='r')
mplt.legend()
mplt.title('Data visualisation')
mplt.show()


# In[36]:


Lr = LinearRegression()
Lr.fit(x_train.values.reshape(-1,1), y_train.values)


# In[37]:


pred = Lr.predict(x_test.values.reshape(-1,1))
pred


# In[38]:


pred1 = pd.DataFrame(pred)
pred1


# In[39]:


mplt.plot(x_test, pred, label = 'Linear Regression', color='b')
mplt.scatter(x_test, y_test, label = 'Testing Data', color='r')
mplt.legend()
mplt.title('Plot test data')
mplt.show()


# In[42]:


# What will be predicted score if a student studies for 9.25 hrs/ day?
Lr.predict([[9.25]])


# In[41]:




