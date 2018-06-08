
# coding: utf-8

# # Exploratory Data Analysis (EDA) on House price prediction dataset
# The goal of this project was to use EDA, visualization, data cleaning, preprocessing, and linear models to predict home prices given the features of the home, and interpret your linear models to find out what features add value to a home.

# !(houseprice predicition)("![image.png](attachment:image.png)")

# !(prediction)("![image.png](attachment:image.png)")

# # Data description
# The data is in csv format.In computing, a comma-separated values (CSV) file stores tabular data (numbers and text) in plain text. Each line of the file is a data record. Each record consists of one or more fields, separated by commas.
# 
# Data are collected on 20 different properties of the house_price,one of which is grade is based on sensory data and rest is on properties of the house including bedrooms,bathrooms etc.

# # Attribute information
# more information of dataset
# 1. id
# 2. date
# 3. price
# 4. bedrooms
# 5. bathrooms
# 6. sqft_living
# 7. sqft_lot
# 8. floors
# 9. waterfront
# 10. view
# 11. grade
# 12. sqft_above
# 13. sqft_basement
# 14. yr_built
# 15. yr_renovated
# 16. zipcode 
# 17. lat 
# 18. long 
# 19. sqft_living15
# 20. sqft_lot15

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
d=pd.read_csv("https://raw.githubusercontent.com/Shreyas3108/house-price-prediction/master/kc_house_data.csv")


# In[19]:


d.head()


# In[21]:


d.tail()


# In[22]:


d.describe()


# In[25]:


print('nullcheck')
d.isnull().sum()


# # univariate analysis
# Uni means one. Univariate means one variable analysis. The key pointers to the Univaraite analysis are to find out the outliers present in the data. We also tend to find the dsitribution of the data on the dataset which can further help us for the Bivaraite/Multivariate analysis.

# ## grade count plot
# some description about grade count plot
# 1. The Grade rating 7 is maximum,grade rating 7 and 8 is high number
# 

# In[23]:


sns.countplot(x='grade',data=d)
sns.despine()


# ##  Price count plot
# 

# In[29]:


sns.countplot(x='price',data=d)
sns.despine()


# In[30]:


sns.boxplot(d['price'])
sns.despine()


# ## sqft_living plot
# 1. We can see that the distribution is skewed towards right.
# 2. There are outliers present in the data.

# In[32]:


sns.distplot(d['sqft_living'])
sns.despine()


# ## bedrooms count plot
# ->some properties
# 1. Bedrooms having 3 and 4 high number
# 2. less outliers

# In[33]:


sns.countplot(x='bedrooms',data=d)
sns.despine()
pl.title('Total no of bedrooms in each house')


# In[53]:


sns.violinplot(x='bedrooms',data=d)
sns.despine()


# In[60]:


sns.boxplot(x='sqft_above',data=d)
sns.despine()


# # Bivariate analysis
# Two variable analysis. We want to find out the relationship between two points. Explore the data.

# In[34]:


d.corr()


# In[36]:


pl.figure(figsize =(13,13))
sns.heatmap(d.corr(),annot=True)
pl.show()


# ## bedrooms vs grade joint plot
# 1. bedrooms and grade having corelation coeeficient is 0.36

# In[55]:


sns.jointplot(x='bedrooms',y='grade',data=d)
pl.title('bedrooms and grade')
sns.despine()


# ## price vs grade
# we find about price vs grade in regplot that
# 1. price and grade having correlation coefficient is 0.67
# 2. we can see that price having greater than 400000 grade lies b/w 10-15
# 3. less no of house in price having greater than 400000

# In[47]:


sns.regplot(x='price',y='grade',data=d,marker='*',scatter_kws={'s':58},color='red')
sns.despine()


# In[ ]:


sns.boxplot(x='price',y='grade',data=d)
sns.despine()


# In[ ]:


pl.hist(x='sqft_above',data=d)
pl.show()


# ## sqft_living vs sqft_above
# 1. sqft_living and sqft_above having correlation coefficient of 0.88

# In[ ]:


sns.regplot(x='sqft_living',y='sqft_above',data=d,x_jitter=0.2,scatter_kws={'alpha':0.2})
sns.despine()
print('hello')


# ## floors vs price 

# In[ ]:


sns.regplot(x='price',y='floors',data=d,scatter_kws={'alpha':0.3})
sns.despine()
pl.title('price vs grade')


# ## bedrooms vs grade countplot

# In[57]:


sns.regplot(x='bedrooms',y='grade',data=d,scatter_kws={'alpha':0.3})
sns.despine()


# In[65]:


sns.boxplot(x = 'grade', y= 'floors',data = d)
sns.pointplot(x='grade',y='floors',data = d)
sns.despine()


# In[63]:


grid = sns.FacetGrid(d, col='grade',col_wrap = 2)
grid.map(pl.scatter,'bedrooms','floors',alpha = 0.2)


# # multivariant analysis 

# In[68]:


sns.lmplot(x = 'bedrooms' , y = 'floors' , hue='grade',data = d ,scatter_kws={'alpha':0.3}, fit_reg = False)
sns.despine()


# #  Final plots
# 1. bedrooms vs grade in bivarant analysis()
# 2. price count plot()
# 3. multivariant analysis of grade,floors and bathrooms(1.>house which have 2 floor having maximum variant of grade)
