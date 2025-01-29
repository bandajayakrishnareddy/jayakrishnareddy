#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv("sss.csv")
print(data)


# In[5]:


print(type(data))
print(data.shape)
print(data.size)


# In[9]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[14]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[15]:


data1[data1.duplicated(keep = False)]


# In[16]:


data1[data1.duplicated()]


# In[ ]:





# In[3]:


sns.violinplot(data=data1, x = "Weather" , y="Ozone", palette="Set2")


# In[5]:


sns.swarmplot(data=data1, x = "Weather", y = "Ozone",color="orange",palette="Set2", size=6)


# In[ ]:


sns.stripplot(data=data1, x "Weather", y = "Ozone", color)


# In[8]:


sns.kdeplot(data=data1["Ozone"], fill=True, color="blue")
sns.rugplot(data=data1["Ozone"], color="yellow")


# In[9]:


sns.boxplot(data = data1, x = "Weather", y="Ozone")


# In[10]:


plt.scatter(data1["Wind"], data1["Temp"])


# In[11]:


data1["Wind"].corr(data1["Temp"])


# In[13]:


data1_numeric = data.iloc[:,[0,1,2,6]]
data1_numeric


# In[ ]:




