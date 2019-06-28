# -*- coding: utf-8 -*-
"""
@author: Omkar Shidore
@github: https://www.github.com/OmkarShidore
"""

#import libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import CSV
USAhousing = pd.read_csv('USA_Housing.csv')

USAhousing.info()

#pairplots
sns.pairplot(USAhousing)

#dist plot for price range in data set
sns.distplot(USAhousing['Price'])

#heatmap of correlated data
sns.heatmap(USAhousing.corr())

#Creating X and y (Verticle Split
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']


#importing train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

#import model : LinearRegression
from sklearn.linear_model import LinearRegression

#instanctiating object of class LinearRegression
lm = LinearRegression()

#training model
lm.fit(X_train,y_train)

#model evaluation
print('intercept: ', lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df

#prediction
predictions = lm.predict(X_test)

#scatter plot to check the accuracy by checking the slope
plt.scatter(y_test,predictions)

#residual histogram
sns.distplot((y_test-predictions),bins=50)

#Regretion Evaluation trhough its metrics 
#Mean Absolute Error (MAE)
#Mean Squared Error  (MSE)
#Root Mean Squared Error (RMSE)

#importing metrics
from sklearn import metrics

#metrics result
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))