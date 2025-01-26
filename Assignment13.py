#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv('Day_13_Pharma_data.csv')
except FileNotFoundError:
    print("Error: File 'Day_13_Pharma_data.csv' not found.")
    exit()

df.drop_duplicates(inplace=True)

required_columns = ['Region', 'Sales', 'Marketing_Spend', 'Effectiveness', 'Age_Group', 'Product', 'Trial_Period']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"Warning: Missing columns in the dataset: {missing_columns}")

if 'Region' in df.columns and 'Sales' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Region', y='Sales', data=df, estimator='sum', errorbar=None)
    plt.title('Total Sales per Region')
    plt.ylabel('Total Sales')
    plt.xlabel('Region')
    plt.xticks(rotation=45)
    plt.show()

if 'Marketing_Spend' in df.columns and 'Sales' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Marketing_Spend', y='Sales', data=df)
    plt.title('Marketing Spend vs Sales')
    plt.xlabel('Marketing Spend')
    plt.ylabel('Sales')
    plt.show()

if 'Age_Group' in df.columns and 'Effectiveness' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Age_Group', y='Effectiveness', data=df)
    plt.title('Drug Effectiveness by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Effectiveness')
    plt.show()

if 'Trial_Period' in df.columns and 'Sales' in df.columns and 'Product' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Trial_Period', y='Sales', hue='Product', data=df, marker='o')
    plt.title('Sales Trend per Product Over Trial Periods')
    plt.xlabel('Trial Period')
    plt.ylabel('Sales')
    plt.show()
elif 'Trial_Period' in df.columns and 'Sales' in df.columns:
    print("Skipping line plot: 'Product' column is missing.")

if all(col in df.columns for col in ['Sales', 'Marketing_Spend', 'Effectiveness']):
    correlation_matrix = df[['Sales', 'Marketing_Spend', 'Effectiveness']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap between Sales, Marketing Spend, and Effectiveness')
    plt.show()
else:
    print("Skipping heatmap: Required columns for correlation are missing.")


# In[ ]:




