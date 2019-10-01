#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ggplot import *
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz


# In[2]:


#set path
os.chdir("/home/zozo/Documents/edwisor/Project/Cab Fare/")


# In[3]:


#check path
os.getcwd()


# In[4]:


#Load Train and Test data
train=pd.read_csv("train_cab.csv")
test=pd.read_csv("test.csv")
train.shape


# In[5]:


test.shape


# In[6]:


train.head()


# In[7]:


#check datatypes of train
train.dtypes


# In[7]:


#convert datatypes
train['fare_amount'].loc[1123]=430
train['fare_amount']=pd.to_numeric(train['fare_amount'])
train['passenger_count']=train['passenger_count'].astype(object)


# ### Missing Value Analysis

# In[8]:


#Create dataframe with missing percentage
missing_val = pd.DataFrame(train.isnull().sum())

#Reset index
missing_val = missing_val.reset_index()

#Rename variable
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})

#Calculate percentage
missing_val['Missing_percentage'] = (missing_val['Missing_percentage']/len(train))*100

#descending order
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)


# In[48]:


missing_val


# In[11]:


#As Passnenger_count is a categorical variable we will use mode to impute it


# In[11]:


train['passenger_count'].value_counts()


# In[9]:


#mode- as 1 has occurred for most no. of times we will replace na with 1 i.e Mode

train['passenger_count'] = train['passenger_count'].fillna(1)


# In[13]:


#Now to impute missing value of fare_amount we will use Mean or Median


# In[10]:


train['fare_amount'].loc[2000]


# In[16]:


#actual value=17.7
#mean=15.0407
#median=8.5
#As Mean is giving better result we will use mean to replace na in fare_amount variable


# In[11]:


#Mean
train['fare_amount'] = train['fare_amount'].fillna(train['fare_amount'].mean())


# In[18]:


#Median
#train['fare_amount'] = train['fare_amount'].fillna(train['fare_amount'].median())


# In[16]:


missing_val


# In[ ]:





# ### Now we will remove outlier location from train data using test dataset

# In[17]:


#Pickup latitude range
print("Range of Pickup Latitude is", (min(train['pickup_latitude']),max(train['pickup_latitude'])))


# In[18]:


#Dropoff latitude range
print("Range of Dropoff Latitude", (min(train['dropoff_latitude']),max(train['dropoff_latitude'])))


# In[19]:


# now lets see what are the boundaries in test dataset 


# In[20]:


#longitude boundary
print("Longitude Boundary in test")
min(test.pickup_longitude.min(), test.dropoff_longitude.min()),max(test.pickup_longitude.max(), test.dropoff_longitude.max())


# In[21]:


#latitude boundary
print("Latitude Boundary in test")
min(test.pickup_latitude.min(), test.pickup_latitude.min()),max(test.pickup_latitude.max(), test.pickup_latitude.max())


# In[12]:


#set boundaries
boundary={'min_longitude':-74.263242,
              'min_latitude':40.573143,
              'max_longitude':-72.986532, 
              'max_latitude':41.709555}


# In[13]:


train.loc[~((train.pickup_longitude >= boundary['min_longitude'] ) & (train.pickup_longitude <= boundary['max_longitude']) &
            (train.pickup_latitude >= boundary['min_latitude']) & (train.pickup_latitude <= boundary['max_latitude']) &
            (train.dropoff_longitude >= boundary['min_longitude']) & (train.dropoff_longitude <= boundary['max_longitude']) &
            (train.dropoff_latitude >=boundary['min_latitude']) & (train.dropoff_latitude <= boundary['max_latitude'])),'outlier']=1
train.loc[((train.pickup_longitude >= boundary['min_longitude'] ) & (train.pickup_longitude <= boundary['max_longitude']) &
            (train.pickup_latitude >= boundary['min_latitude']) & (train.pickup_latitude <= boundary['max_latitude']) &
            (train.dropoff_longitude >= boundary['min_longitude']) & (train.dropoff_longitude <= boundary['max_longitude']) &
            (train.dropoff_latitude >=boundary['min_latitude']) & (train.dropoff_latitude <= boundary['max_latitude'])),'outlier']=0

# Let us drop outlier locations
train=train.loc[train['outlier']==0]
train.drop(['outlier'],axis=1,inplace=True)


# In[14]:


train.shape


# ### Now let us extract important features from pickup_datetime

# In[15]:


#lets create a function to get important features from pickup_datetime variable in train and test datasets
def clean(data):
    data['pickup_datetime']=data.pickup_datetime.str.slice(-23,-3)
    data['pickup_datetime']=pd.to_datetime(data.pickup_datetime)
    data['day'] = data['pickup_datetime'].dt.day
    data['year'] = data['pickup_datetime'].dt.year 
    data['month'] = data['pickup_datetime'].dt.month 
    data['hour'] = data['pickup_datetime'].dt.hour 
    data['weekday'] = data['pickup_datetime'].dt.weekday
    data=data.dropna(subset=['year','month','hour','weekday'])
    return data


# In[16]:


train=clean(train)


# In[17]:


train.shape


# In[18]:


test=clean(test)


# In[19]:


test.shape


# In[ ]:





# ### Now let's calculate trip distance from picup and dropoff latitude and longitude

# In[20]:


def trip_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  
    return km


# In[21]:


train['trip_distance']=trip_distance(train['pickup_longitude'],train['pickup_latitude'],
                                     train['dropoff_longitude'],train['dropoff_latitude'])


# In[22]:


train.shape


# In[23]:


test['trip_distance']=trip_distance(test['pickup_longitude'],test['pickup_latitude'],
                                     test['dropoff_longitude'],test['dropoff_latitude'])


# In[24]:


test.shape


# In[29]:


train.head()


# In[ ]:





# In[36]:


#Now look at the summary of the data
train.describe()


# #### 1.look at the summary of fare_amount,passenger_count and trip_distance
# #### 2.fare_amount has minimun value as negative value which is not possible so we will drop negative values
# #### 3.passenger_count has minimum value 0 which also does not have any significant value for our model so we will drop it
# #### 4.trip_distance also has minimum value as 0 which is also of no use
# 

# In[40]:


#let's clean fare_amount variable


# In[25]:


#Removing all the fares having value zero
train.drop(train[train['fare_amount'] < 1].index, inplace = True)


# In[26]:


#Removing all the passenger_counts having value zero or more than six
train=train[train['passenger_count']<=6]
train=train[train['passenger_count']>=1]


# In[27]:


#lets check values of passenger_counts
train['passenger_count'].value_counts()


# ### there cannot be 1.3 passenger so lets remove it

# In[28]:


train=train[train['passenger_count']!=1.3]


# In[29]:


##Now remove the trip_distance having value less than 0.2 as most of people will not take a cab for distance below 200 meters
train = train.loc[train['trip_distance'] >=0.2]


# In[68]:


train['trip_distance'].value_counts()


# In[ ]:





# In[43]:


train.describe()


# ###Now if you will carefully look at the summary of data you will find out that max value in fare_amount is 54343
# which is way more than the mean value which is 15.19 that means this is an outlier. Same is the case with trip_distance where mean is 3.4 and max value is 101.09 so lets remove outliers from data

# ###  outliers

# In[48]:


#first look at the relation between trip_distance and fare_amount


# In[36]:


plt.scatter(x=train['trip_distance'],y=train['fare_amount'])
plt.xlabel("Trip Distance")
plt.ylabel("Fare Amount")
plt.ylim(0, 500)
plt.title("Trip Distance vs Fare Amount")


# #### As you can see in scatterplot that fare amount is almost fixed for trip over 80 kM and sometimes fares are very high for very short distance which is basically an outlier.

# In[37]:


# #Plot boxplot to visualize Outliers
get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(train['fare_amount'])


# ###### Now after looking the scatter plot of Fare vs Trip_distance and boxplot of Fare amount we will remove all the fares having value more than 150

# In[30]:


train=train[train['fare_amount']<150]


# ###### Now lets draw boxplot for trip_distance for train and test datasets

# In[47]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(train['trip_distance'])


# In[48]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(test['trip_distance'])


# ##### After visualising boxplot for trip_distance of train and test datasets we know that they contain almost same trip distance so we will leave trip distance as it is.

# 

# In[31]:


train.shape


# In[50]:


train.describe()


# In[56]:


#now you can see after removing ouliers max value and mean of fare_amount has been changed


# In[44]:


# train.to_csv("trained_data.csv", index=False)


# In[45]:


# test.to_csv("processed_test_data.csv",index=False)


# 

# In[51]:


#let's visualize fare_amount
plt.figure(figsize = (14, 4))
n, bins, patches = plt.hist(train.fare_amount, 50, facecolor='green', alpha=0.8)
plt.xlabel('Fare Amount')
plt.title('Histogram of Fare Amount')
plt.xlim(0, 100)
plt.show();


# In[60]:


#as we can see in the histogram most Fare drops between 5 to 15 dollars indicating short trips


# In[52]:


#now lets visualize passenger_count
train['passenger_count'].value_counts().plot.bar(color = 'blue', edgecolor = 'black');
plt.title('Bargraph of passenger counts'); plt.xlabel('Passenger counts'); plt.ylabel('Count');


# In[62]:


#you can see in the above bargraph that most of the time single passengers have booked cab and family booking is least


# In[53]:


#lets visualize the trip_distance\
plt.figure(figsize = (14, 4))
n, bins, patches = plt.hist(train.trip_distance, 200, facecolor='blue', alpha=0.75)
plt.xlabel('trip_distance')
plt.xlim(0, 40)
plt.title('Histogram of trip distance')
plt.show();


# In[64]:


#in the above histogram we can see that maximum no. of times people took short trips between somewhere 0.7 to 3 KM


# In[39]:


train.shape


# In[32]:


def modeling(data,target,drop_cols,split=0.25):
    new_data=data.drop(drop_cols,axis=1)
    X=new_data.drop([target],axis=1)
    y=new_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split,random_state=42)
    print("train shape",X_train.shape)
    print("test shape",X_test.shape)
    return X_train, X_test, y_train, y_test
    return new_data


# In[33]:


X_train, X_test, y_train, y_test=modeling(train,'fare_amount',drop_cols=['pickup_datetime'],split=0.2)


# In[57]:


train.dtypes


# In[58]:


train.head()


# In[59]:


test.dtypes


# ## Model Development

# #### Linear Regression

# In[34]:


# Train the model using the training sets
model1 = sm.OLS(y_train, X_train).fit()


# In[35]:


model1.summary()


# In[36]:


y_pred1=model1.predict(X_test)


# #### Lets use RMSE to test accuracy of the Model

# In[37]:


lm_rmse=np.sqrt(mean_squared_error(y_pred1, y_test))
print("RMSE for Linear Regression is ",lm_rmse)


# ##### Now lets try another linear regression by changing split

# In[39]:


X_train1, X_test1, y_train1, y_test1=modeling(train,'fare_amount',drop_cols=['pickup_datetime'],split=0.3)


# In[40]:


# Train the model using the training sets
model2 = sm.OLS(y_train1, X_train1).fit()


# In[41]:


model2.summary()


# In[42]:


y_pred2=model2.predict(X_test1)


# In[43]:


lm_rmse=np.sqrt(mean_squared_error(y_pred2, y_test1))
print("RMSE for Linear Regression is ",lm_rmse)


# In[ ]:





# #### DecisionTree

# In[44]:


#Decision tree for regression
fit_DT = DecisionTreeRegressor(max_depth=6,random_state=42).fit(X_train, y_train)


# In[45]:


predictions_DT = fit_DT.predict(X_test)


# #### Lets use RMSE to test accuracy of the Model

# In[46]:


dt_rmse=np.sqrt(mean_squared_error(predictions_DT,y_test))
print("RMSE = ",dt_rmse)


# In[ ]:





# #### Random Forest

# In[47]:


fit_RF = RandomForestRegressor(n_estimators = 50,random_state=42).fit(X_train,y_train)


# In[48]:


prediction_RF=fit_RF.predict(X_test)


# #### Lets use RMSE to test accuracy of the Model

# In[49]:


rf_rmse=np.sqrt(mean_squared_error(prediction_RF,y_test))
print("RMSE = ",rf_rmse)


# In[ ]:





# ### As we got best Accuracy with RandomForest Model we will use this Model to predict Fare

# In[75]:


test.describe()


# In[76]:


test.head()


# In[77]:


test=test.drop(['pickup_datetime'], axis=1)


# In[78]:


test.shape


# In[79]:


predicted_fare=fit_RF.predict(test)


# In[80]:


test['predicted_fare']=predicted_fare


# In[81]:


test.head(10)


# In[82]:


test.to_csv("test_predicted.csv",index=False)


# In[ ]:




