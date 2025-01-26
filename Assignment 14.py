#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('Day_14_Pharma_data.csv')

# Ensure relevant columns are numeric
data['Effectiveness'] = pd.to_numeric(data['Effectiveness'], errors='coerce')
data['Side_Effects'] = pd.to_numeric(data['Side_Effects'], errors='coerce')
data['Marketing_Spend'] = pd.to_numeric(data['Marketing_Spend'], errors='coerce')

# Drop rows with missing or invalid values
data_cleaned = data.dropna(subset=['Effectiveness', 'Side_Effects', 'Marketing_Spend'])


# In[13]:


effectiveness_by_region = data_cleaned.groupby(['Product_Name', 'Region'])['Effectiveness'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Region', y='Effectiveness', hue='Product_Name', data=effectiveness_by_region, palette='viridis')
plt.title('Average Drug Effectiveness by Product and Region', fontsize=16)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Average Effectiveness', fontsize=12)
plt.legend(title='Product Name')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[14]:


plt.figure(figsize=(14, 6))

# Effectiveness distribution
plt.subplot(1, 2, 1)
sns.violinplot(x='Product_Name', y='Effectiveness', data=data_cleaned, palette='muted')
plt.title('Effectiveness Distribution by Product', fontsize=14)
plt.xlabel('Product Name', fontsize=12)
plt.ylabel('Effectiveness', fontsize=12)
plt.xticks(rotation=45)

# Side Effects distribution
plt.subplot(1, 2, 2)
sns.violinplot(x='Product_Name', y='Side_Effects', data=data_cleaned, palette='muted')
plt.title('Side Effects Distribution by Product', fontsize=14)
plt.xlabel('Product Name', fontsize=12)
plt.ylabel('Side Effects', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# In[15]:


sns.pairplot(data_cleaned, vars=['Effectiveness', 'Side_Effects', 'Marketing_Spend'], hue='Product_Name', palette='tab10', diag_kind='kde')
plt.suptitle('Pairplot of Effectiveness, Side Effects, and Marketing Spend', y=1.02, fontsize=16)
plt.show()


# In[16]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='Trial_Period', y='Effectiveness', data=data_cleaned, palette='coolwarm')
plt.title('Effectiveness by Trial Period', fontsize=16)
plt.xlabel('Trial Period', fontsize=12)
plt.ylabel('Effectiveness', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[17]:


plt.figure(figsize=(8, 6))
sns.regplot(x='Marketing_Spend', y='Effectiveness', data=data_cleaned, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
plt.title('Marketing Spend vs. Effectiveness', fontsize=16)
plt.xlabel('Marketing Spend', fontsize=12)
plt.ylabel('Effectiveness', fontsize=12)
plt.tight_layout()
plt.show()


# In[ ]:




