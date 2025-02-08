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


# In[22]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model2,alpha=.05)
y = [i for i in range(-2,8)]
x = [leverage_cutoff for i in range(10)]
plt.plot(x,y,'r+')
plt.show()


# #### Observation 
# - From the above plot, it is evident that data points 65,70,76,78,79,80 are the influence.
# - As their H leverage values are higher and size is higher

# In[23]:


cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)


# In[24]:


cars2


# #### Bulid Model3 on cars2 dataset

# In[25]:


model3 = smf.ols('MPG~VOL+SP+HP',data = cars2).fit()


# In[26]:


model3.summary()


# #### preformance Metrics for model3

# In[27]:


df3= pd.DataFrame()
df3["actual_y3"] =cars2["MPG"]
df3.head()


# In[28]:


pred_y3 = model3.predict(cars2.iloc[:,0:3])
df3["pred_y3"] = pred_y3
df3.head()


# In[29]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3["actual_y3"], df3["pred_y3"])
print("MSE :", mse )
print("RMSE :",np.sqrt(mse))


# #### Comparison of models
#                      
# 
# | Metric         | Model 1 | Model 2 | Model 3 |
# |----------------|---------|---------|---------|
# | R-squared      | 0.771   | 0.770   | 0.885   |
# | Adj. R-squared | 0.758   | 0.761   | 0.880   |
# | MSE            | 18.89   | 18.91   | 8.68    |
# | RMSE           | 4.34    | 4.34    | 2.94    |
# 
# 
# - **From the above comparison table it is observed that model3 is the best among all with superior performance metrics**

# #### Check the validity of model assumptions

# In[30]:


model3.resid


# In[36]:


import statsmodels.api as sm
qqplot=sm.qqplot(model3.resid,line='q')
plt.title("Normal q-q plot of reiiduals")
plt.show()


# In[35]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()



plt.figure(figsize=(6,4))
plt.scatter(get_standardized_values(model3.fittedvalues),
            get_standardized_values(model3.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[ ]:




