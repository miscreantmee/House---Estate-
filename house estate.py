#!/usr/bin/env python
# coding: utf-8

# # House Estate Price Predictor

# In[1]:


import pandas as pd


# In[2]:


Housing = pd.read_csv("data.csv")


# In[3]:


Housing.head()


# In[4]:


Housing.info()


# In[5]:


Housing['CHAS'].value_counts()


# In[6]:


Housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


Housing.hist(bins=50,figsize=(20,15))


# # Train-Test Splitting
# 

# In[10]:


#forlearning purpose
import numpy as np
def split_train_test(data,test_ratio):
  shuffled= np.random.permutation(len(data))
  np.random.seed(42)
  print(shuffled)  
  test_set_size = int(len(data) * test_ratio)
  test_indices = shuffled[:test_set_size]
  train_indices = shuffled[test_set_size:]
  return data.iloc[train_indices],data.iloc[test_indices]


# In[11]:


#train_set , test_set = split_train_test(Housing , 0.2)


# In[12]:


#print(f"Rows in train set :{len(train_set)}\nRows in test set :{len(test_set)}\n")


# In[13]:


from sklearn.model_selection import train_test_split
train_set ,test_set=train_test_split(Housing, test_size =0.2,random_state=42)


# In[14]:


print(f"Rows in train set :{len(train_set)}\nRows in test set :{len(test_set)}\n")


# In[15]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2 ,random_state=42)
for train_index  , test_index in split.split(Housing,Housing['CHAS']):
    strat_train_set = Housing.loc[train_index] 
    strat_test_set = Housing.loc[test_index]


# In[16]:


strat_test_set.describe()


# In[17]:


Housing= strat_train_set.copy()


# 
# # loooking for corelations

# In[18]:


corr_matrix=Housing.corr()


# In[19]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[20]:


from pandas.plotting import scatter_matrix
attributes = ['RM','ZN','B','MEDV','LSTAT']
scatter_matrix(Housing[attributes] , figsize=(12,8))


# In[21]:


Housing = strat_train_set.drop('MEDV', axis=1)
Housing_labels = strat_train_set['MEDV'].copy()


# 
# # Imputation

# In[22]:


from sklearn.impute import SimpleImputer
imputer= SimpleImputer(strategy='median')
imputer.fit(Housing)


# In[23]:


imputer.statistics_


# In[24]:


X=imputer.transform(Housing)


# In[25]:


Housing_tr= pd.DataFrame(X,columns=Housing.columns)


# In[26]:


Housing_tr.describe()


# In[ ]:





# # Creating pipeline

# In[27]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[28]:


my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])


# In[29]:


Housing_num_tr = my_pipeline.fit_transform(Housing)


# In[30]:


Housing_num_tr


# # Selecting a desired model

# In[31]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model=  RandomForestRegressor()
model.fit(Housing_num_tr , Housing_labels)


# In[32]:


some_data = Housing.iloc[:5]


# In[33]:


some_labels = Housing_labels[:5]


# In[34]:


prepared_data= my_pipeline.transform(some_data)


# In[35]:


model.predict(prepared_data)


# In[36]:


list(some_labels)


# # evaluating the model

# In[37]:


from sklearn.metrics import mean_squared_error
housing_predictions =model.predict(Housing_num_tr)
mse= mean_squared_error(Housing_labels,housing_predictions)
rmse = np.sqrt(mse)


# In[38]:


rmse


# # using better evaluation techniques

# In[39]:


from sklearn.model_selection import cross_val_score
scores =cross_val_score(model ,Housing_num_tr, Housing_labels, scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)


# In[40]:


rmse_scores


# In[41]:


def print_scores(scores):
    print('Scores: ', scores)
    print('Mean: ',scores.mean())
    print('Standard Deviation: ',scores.std())


# In[42]:


print_scores(rmse_scores)


# # SAVING THE MODEL

# In[43]:


from joblib import dump, load
dump(model, 'House Estate')


# # TESTING THE MODEL

# In[49]:


X_test = strat_test_set.drop('MEDV',axis=1)
Y_test= strat_test_set['MEDV'].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse= mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
#print(final_predictions,list(Y_test))


# In[50]:


final_rmse


# In[ ]:





# In[ ]:




