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

#read csv
train = pd.read_csv('titanic_train.csv')   

train.head()

#check for empty vars
train.isnull()

#heatmap of missing values
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#count plot of the survived and deceased
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')

#count plot of the survived and deceased using hue='sex'
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')

#count plot of survived and deceased as per hue='seating class'
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')

#imputing missing age
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
    
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#too many missing values in col cabin, so lets drop it
train.drop('Cabin',axis=1,inplace=True)

#using get dummis for col sex as our model will just drop those cols if not used as features
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train = pd.concat([train,sex,embark],axis=1)
train.head()
    
#train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)

#Training
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

#predections
predictions = logmodel.predict(X_test)

#evaluation
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))