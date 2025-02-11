#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[2]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[3]:


Univ.info()


# In[4]:


Univ.isna().sum()


# In[5]:


Univ.describe()


# In[9]:


univ1 = Univ.iloc[:,1:]
univ1


# In[12]:


cols = univ1.columns


# In[19]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_univ_df = pd.DataFrame(scaler.fit_transform(univ1),columns = cols )
scaled_univ_df


# In[22]:


from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_univ_df)


# In[23]:


clusters_new.labels_


# In[24]:


set(clusters_new.labels_)


# In[28]:


Univ['clusterid_new'] = clusters_new.labels_


# In[29]:


Univ[Univ['clusterid_new']==1]


# In[30]:


Univ.iloc[:,1:].groupby("clusterid_new").mean()


# #### Observation
# - Cluster 2 appers to be the top rated universities cluster as the cut off score top10, SFRatio parameter mean values are highest 
# - Cluster 1 appers to occupy the middle level rated universities
# - Cluster 0 comes as the lower level rated universities

# In[31]:


Univ[Univ['clusterid_new']==0]


# In[34]:


wcss = []
for i in range(1, 20):
    Kmeans = KMeans(n_clusters=i,random_state=0 )
    Kmeans.fit(scaled_univ_df)
    wcss.append(Kmeans.inertia_)
print(wcss)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of cluster')
plt.ylabel('WCSS')
plt.show()


# In[ ]:




