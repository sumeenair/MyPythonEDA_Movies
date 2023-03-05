#!/usr/bin/env python
# coding: utf-8

# In[25]:


# First let's import the packages we will use in this project
# You can do this all now or as you need them
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None



# Now we need to read in the data
df = pd.read_csv('C:/Users/sumee/OneDrive/Documents/Personal/OneDrive/Personal Projects/Pyhton Project - Dataset/movies.csv')


# In[26]:


df


# In[27]:


# Check for missing data - Lots of ways to do this, one of them is this below : 
# Loop through the data and see if there is anything missing in percent 

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# In[7]:


# Data Types for our columns

print(df.dtypes)


# In[23]:


## create correct year column 
#df['year'] = df['released'].astype(str).str[9:13]
#df.head()


# In[30]:


# Order by gross revenue 
df.sort_values(by=['gross'], inplace=False, ascending = False)


# In[29]:


pd.set_option('display.max_rows', None)


# In[31]:


#Drop any duplicates if any ( to see is there is issues in the quality of data)
df['company'].drop_duplicates().sort_values(ascending = False)


# In[8]:


# Are there any Outliers?

df.boxplot(column=['gross'])


# In[9]:


df.drop_duplicates()


# In[33]:


# Order our Data a little bit to see

df = df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[ ]:


## Assumption is that : more budget will bring in more revenue , more known the company is - more revenue ? Lets check this 


# In[36]:


# 1. Lets do a scatter plot with gross vs budget 
#sns.regplot(x="gross", y="budget", data=df)
plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for movies')
plt.show


# In[41]:


# Lets do a seaborn plot to see the budget vs gross
sns.regplot(x="budget", y="gross", data=df, scatter_kws={"color":"red"}, line_kws={"color":"black"})


# In[13]:


# Lets start looking at correlation 
#Correlation Matrix between all numeric columns ( only works on numercial)
#Person is default , kendall, spearman ( slightly different corr)

df.corr(method ='pearson')


# In[14]:


df.corr(method ='kendall')


# In[15]:


df.corr(method ='spearman')


# In[16]:


correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Numeric Features")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[42]:


df_numerized = df

for col_name in df_numerized.columns: 
    if(df_numerized[col_name].dtype == 'object'): 
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
        
df_numerized


# In[44]:


correlation_matrix = df_numerized.corr()

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Numeric Features")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[45]:


high_corr = sorted_pairs[(sorted_pairs)> 0.5]


# In[ ]:




