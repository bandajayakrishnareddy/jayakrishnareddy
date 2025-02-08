#!/usr/bin/env python
# coding: utf-8

# #### Assumption in Multilinear Regression
# -Linearity: The relationship between the predictors and the response is linear
# -independence:Observation are independent of each other
# -Homoscedasticity: The residuals (differences between observed and predicted values) exconstant varince at all levels of the predictor
# 

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


cars = pd.DataFrame(cars, columns=["HP", "VOL", "SP", "WT", "MPG"])
cars.head()


# #### Description of columns
# - MPG: Milege of the car (Mile per Gallon)
# - HP: Horese Power of the car
# - VOL: Volume of the car(size)
# -SP: top speed of thr car
# -WT: weight of the car

# In[4]:


cars.isna().sum()


# #### Observation
# - There are no missing values
# -There are 81 observation
# -The data types of the columns are relevant and valid

# #### Observation from boxplot and histograms
# - There are some extreme values (outliers) observed in towards the right rail of sp and hp distributiond
# - In VOL and WT columns, a few outliers are observed in both tails of their distributions
# - The extreme values of cars data may have come from the specially designed nature of cars
# -AS this multi_dimensional data,

# In[5]:


cars[cars.duplicated()]


# In[6]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[7]:


cars.corr()


# #### Observations
# - highest positive correlation between WT and VOL
# - higher postive correlation between HP and SP 
# - most of the correlations are negative correlation 
# - between x and y Hp and MPG has the highest correlation

# In[8]:


model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
model1.summary()


# #### OBSERAVTION FROM MODEL SUMMARY
# - THE r-SQUARED AND ADJUSTED r-SUARED VALUES ARE GOOD AND ABOUT 75% OF VARIABILITY IN y IS EXPLAINED BY x COLUMNS
# - The probabilaty value with respect to F-statistic is close to zero, indicating that all or someof X columns are significant
# - The P-value for VOL and WT are higher than 5% indicaring some interction issue among themselves which ned to be fruther expored

# In[9]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[10]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[11]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# In[12]:


cars.head()


# In[13]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# #### Observation:
# - The ideal range of VIF values shall be between 0 to 10 .However slightly higher values can be tolerated
# - As seen from the very high VIF values for VOL and WT.it is clear that they are prone to multicollinearity promblem
# - Hence it is decited to declear to drop one of columns(eighter VOL or WT) to overcome the multicollinearity
# - it is decided to drop WT and VOL column in further models

# In[14]:


cars1 = cars.drop("WT",axis=1)
cars1.head()


# In[15]:


import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()


# In[16]:


model2.summary()


# #### PREFORMANCE METRICS FOR MODEL2

# In[17]:


df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[18]:


pred_y2 = model2.predict(cars1.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[19]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# #### Observation from model2 summary()
# - The adjusted R-squered value imporoved slightly to 0.76
# - All the P-values for model parameters are less 5% hence they are significant
# - Therefore the HP,VOL,SP columns are finalized as the significant predictor for the MPg
# - There is no improvement MSE value

# #### Identification of high influence points (spatial outliers)

# In[20]:


cars1.shape


# In[21]:


k = 3
n = 81
leverage_cutoff = 3*((k + 1) / n)
leverage_cutoff


# In[24]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model2,alpha=.05)
y = [i for i in range(-2,8)]
x = [leverage_cutoff for i in range(10)]
plt.plot(x,y,'r+')
plt.show()


# #### Observation 
# - From the above plot, it is evident that data points 65,70,76,78,79,80 are the influence.
# - As their H leverage values are higher and size is higher

# In[25]:


cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)


# In[26]:


cars2


# #### Bulid Model3 on cars2 dataset

# model3 = smf.ols('MP

# In[ ]:




