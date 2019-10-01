setwd("/home/zozo/Documents/edwisor/Project/Cab Fare")
getwd()

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

## Read the data
train = read.csv("train_cab.csv", header = T)
test=read.csv("test.csv",header = T)

##summary of data
summary(train)
dim(train)
sapply(train, typeof)

### convert datatypes
train[,1] = as.numeric(as.character(train[,1]))
train$passenger_count=as.factor(train$passenger_count)


#####Missing Value Analysis###########
missing_val = data.frame(apply(train,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(train)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]

ggplot(data = missing_val[1:3,], aes(x=reorder(Columns, -Missing_percentage),y = Missing_percentage))+
     geom_bar(stat = "identity",fill = "red")+xlab("Parameter")+
     ggtitle("Missing data percentage (Train)") + theme_bw()

##As passenger_count is categorical variable we will impute it using mode
###Mode Method
train$passenger_count[is.na(train$passenger_count)] =as.data.frame(mode(train$passenger_count))

#df=train
train=df

#actual value=17.7
#mean=15.014
#median=8.5
train[2001,1]

## we will impute missing value of fare_amount by using mean or median method
####Mean Method
train$fare_amount[is.na(train$fare_amount)] = mean(train$fare_amount, na.rm = T)

####Median Method
# train$fare_amount[is.na(train$fare_amount)] = median(train$fare_amount, na.rm = T)


### Now we will remove outlier location from train data using test dataset

#Pickup latitude range
 min(train$pickup_latitude)
 max(train$pickup_latitude)

 #dropoff latitiude range
 min(train$dropoff_latitude)
 max(train$dropoff_latitude)
 
 #Pickup longitude range
 min(train$pickup_longitude)
 max(train$pickup_longitude)
 
 #dropoff longitude range
 min(train$dropoff_longitude)
 max(train$dropoff_longitude)
 
 # now lets see what are the boundaries in test dataset 
 
 #longitude boundary
 min(test$pickup_longitude, test$dropoff_longitude)
 max(test$pickup_longitude, test$dropoff_longitude)
 # longitude boundary=(-74.26324,-72.98653)
 
 #latitude boundary
 min(test$pickup_latitude, test$dropoff_latitude)
 max(test$pickup_latitude, test$dropoff_latitude)
 #latitude boundary=(40.56897,41.70956)
 
#set boundaries
min_longitude=-74.26324
min_latitude=40.56897
max_longitude=-72.98653
max_latitude=41.70956

train=subset(train, pickup_longitude >= min_longitude)
train=subset(train,pickup_longitude <= max_longitude)
train=subset(train,pickup_latitude >= min_latitude)
train=subset(train,pickup_latitude<=max_latitude)

train=subset(train, dropoff_longitude >= min_longitude)
train=subset(train,dropoff_longitude <= max_longitude)
train=subset(train,dropoff_latitude >= min_latitude)
train=subset(train,dropoff_latitude<=max_latitude)

df=train
df1=test
train=df
#lets create a function to get important features from pickup_datetime variable in train and test datasets
clean=function(data)
{
  data$week_days = weekdays(as.Date(data$pickup_datetime))
  data %>% separate(pickup_datetime, 
                  c("Year","Month", "Day","hour"))
}

train=clean(train)
train=na.omit(train)
train$week_days=sapply(as.character(train$week_days), switch, "Monday" = 1, "Tuesday" = 2, "Wednesday" = 3, "Thursday" = 4,
                       "Friday"=5,"Saturday"=6,"Sunday"=7,USE.NAMES = F)


test=clean(test)
test$week_days=sapply(as.character(test$week_days), switch, "Monday" = 1, "Tuesday" = 2, "Wednesday" = 3, "Thursday" = 4,
                       "Friday"=5,"Saturday"=6,"Sunday"=7,USE.NAMES = F)


### Now let's calculate trip distance from picup and dropoff latitude and longitude
## Haversine
trip_distance = function(lon1, lat1, lon2, lat2){
  # convert decimal degrees to radians
  lon1 = lon1 * pi / 180
  lon2 = lon2 * pi / 180
  lat1 = lat1 * pi / 180
  lat2 = lat2 * pi / 180
  # haversine formula
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
  c = 2 * atan2(sqrt(a), sqrt(1-a))
  km = 6367 * c
  return(km)
}

## Calculating trip_distance for train data
train$trip_distance=trip_distance(train$pickup_longitude,train$pickup_latitude,
                                  train$dropoff_longitude,train$dropoff_latitude)
 
 
## Calculating trip_distance for test data
test$trip_distance=trip_distance(test$pickup_longitude,test$pickup_latitude,
                                  test$dropoff_longitude,test$dropoff_latitude)

## now lets look at the summary of data 
summary(train)
 
#### 1.look at the summary of fare_amount,passenger_count and trip_distance
#### 2.fare_amount has minimun value as negative value which is not possible so we will drop negative values
#### 3.trip_distance also has minimum value as 0 which is also of no use
#### 4.passenger_count has values other than the 1 to 6 so we will remove it

#let's clean fare_amount variable
 #Removing all the fares having value zero
train=subset(train, fare_amount >= 1)

 #Removing all the passenger_counts having value other than one to six
train$passenger_count <- as.numeric(as.character(train$passenger_count))
train=subset(train, passenger_count >= 1 & passenger_count <= 6)
train$passenger_count=as.factor(train$passenger_count)

#lets check values of passenger_counts
table(train$passenger_count)
### there cannot be 1.3 passenger so lets remove it
train=subset(train, passenger_count != 1.3)
length(unique(train$passenger_count))

##Now remove the trip_distance having value less than 0.2 as most of people will not take 
## a cab for distance below 200 meters
train=subset(train, trip_distance>=0.2)


### Now Once again look at the summary of the data
summary(train)

### Now if you will carefully look at the summary of data you will find out that max value in fare_amount is 54343
### which is way more than the mean value which is 15.17 that means this is an outlier. 
### Same is the case with trip_distance where mean is 3.43 and max value is 101.03 so lets remove outliers from data 
 
############# OUTLIERS #####################

#first look at the relation between trip_distance and fare_amount
library("scales") 
ggplot(train, aes_string(x = train$trip_distance, y = train$fare_amount)) + 
  geom_point(colour ="blue") +
  theme_bw()+ ylab("Fare Amount") + xlab("Trip Distance") + ggtitle("Trip Distance vs Fare Amount") + 
  theme(text=element_text(size=10))+ scale_x_continuous(limits = c(0, 100)) + 
  scale_y_continuous(limits = c(0, 500))

#### As you can see in scatterplot that fare amount is almost fixed for trip over 80 kM and sometimes fares are 
### very high for very short distance which is basically an outlier.

# #Plot boxplot to visualize Outliers
#Box plot
qplot(y=train$fare_amount, x= 1, geom = "boxplot",outlier.colour='red')

#### Now after looking the scatter plot of Fare vs Trip_distance and boxplot of Fare amount 
#### we will remove all the fares having value more than 150

train=subset(train, fare_amount < 150)

###### Now lets draw boxplot for trip_distance for train and test datasets

qplot(y=train$trip_distance, x= 1, geom = "boxplot",outlier.colour='red')

qplot(y=test$trip_distance, x= 1, geom = "boxplot",outlier.colour='red')

##### After visualising boxplot for trip_distance of train and test datasets we know that they contain 
##### almost same trip distance so we will leave trip distance as it is. 

# once again have a look at the summary of the data

summary(train)
#now you can see after removing ouliers max value and mean of fare_amount has been changed
 
#let's visualize fare_amount
ggplot(train, aes_string(x = train$fare_amount)) + 
  geom_histogram(fill="green", colour = "black") + geom_density() +
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  scale_x_continuous(breaks=pretty_breaks(n=10))+
  theme_bw() + xlab("Fare Amount") + ylab("Frequency") + ggtitle("Histogram of Fare Amount") +
  theme(text=element_text(size=20))+ scale_x_continuous(limits = c(0, 100))
 
# as we can see in the histogram most Fare drops between 5 to 15 dollars indicating short trips
 
### now lets visualize passenger_count
ggplot(train, aes_string(x = train$passenger_count)) +
  geom_bar(stat="count",fill =  "blue") + theme_bw() +
  xlab("Passenger Count") + ylab('Count') + scale_y_continuous(breaks=pretty_breaks(n=10)) +
  ggtitle("Marketing Campaign Analysis") +  theme(text=element_text(size=10))
# you can see in the above bargraph that most of the time single passengers have booked cab and 
# family booking is least

# lets visualize the trip_distance
ggplot(train, aes_string(x = train$trip_distance)) + 
  geom_histogram(fill="green", colour = "black") + geom_density() +
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  scale_x_continuous(breaks=pretty_breaks(n=10))+
  theme_bw() + xlab("Fare Amount") + ylab("Frequency") + ggtitle("Histogram of Fare Amount") +
  theme(text=element_text(size=15))
# in this histogram we can see that maximum no. of times people took short trips below somewhere 5KM

############### MODELING #########################
#Divide the data into train and test
set.seed(123)
train_index = sample(1:nrow(train), 0.8 * nrow(train))
train1= train[train_index,]
test1 = train[-train_index,]

### Linear Regression
fit_LM = lm(fare_amount ~ ., data = train1)
#Predict for new test cases
predictions_LM = predict(fit_LM, test1[,-1])

#### Lets use RMSE to test accuracy of the Model
RMSE = function(y, yhat){
  sqrt(mean((y - yhat)^2))
}

RMSE(test1[,1],predictions_LM)
## Accuracy=94.651628
## Error Rate=5.348372


### Decision Tree Regression
# ##rpart for regression
fit_DT = rpart(fare_amount ~ ., data = train1, method = "anova")
predictions_DT = predict(fit_DT, test1[,-1])

#### Lets use RMSE to test accuracy of the Model
RMSE = function(y, yhat){
  sqrt(mean((y - yhat)^2))
}

RMSE(test1[,1],predictions_DT)

## Accuracy=95.560735
## Error Rate=4.439265

### Random Forest Regression

fit_RF <- randomForest(fare_amount ~ .,data    = train1)
predictions_RF=predict(fit_RF,test1[,-1])

#### Lets use RMSE to test accuracy of the Model
RMSE = function(y, yhat){
  sqrt(mean((y - yhat)^2))
}

RMSE(test1[,1],predictions_RF)

### Accuracy=96.333269
### Error Rate=3.666731
 

### As we got best Accuracy with RandomForest Model we will use this Model to predict Fare
summary(test)

### Lets equalize classes of training and test set. Bind the first row of training set 
### to the test set and than delete it
test = rbind(train[1,2:12] , test)
test = test[-1,]
predicted_Fare=predict(fit_RF,test)

test$Predicted_fare=predicted_Fare

write.csv(train, "train_R.csv", row.names = F)
write.csv(test, "predicted_test_R.csv", row.names = F)
