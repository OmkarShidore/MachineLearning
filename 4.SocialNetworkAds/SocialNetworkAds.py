#import libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline

#get data
dataset = pd.read_csv('Social_Network_Ads.csv')

#checking out data
dataset.head()

#data info
dataset.info()

#checking for null values
sns.heatmap(dataset.isnull())

#droping userIID
dataset.drop(['User ID'], axis=1, inplace=True)

#getting dummies for gender
gender=pd.get_dummies(dataset['Gender'],drop_first=True)

#concating gender dataframe with dataset
dataset=pd.concat([dataset,gender],axis=1)

#drop Gender
dataset.drop(['Gender'],inplace=True,axis=1)

dataset.head()

#correlation
sns.heatmap(dataset.corr())

#stardazing variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(dataset.drop('Purchased',axis=1))
scaled_features = scaler.transform(dataset.drop('Purchased',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=['Age', 'EstimatedSalary','Male'])
df_feat.head()

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_feat,dataset['Purchased'],
                                                    test_size=0.30)

#import model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

#training
classifier.fit(X_train, y_train)

#predicting
y_pred = classifier.predict(X_test)

#evaluation
from sklearn.metrics import classification_report,confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cr=classification_report(y_test,y_pred)
print("Confusion Matrix: \n",cm)
print("Classification Report: \n", cr)

print("Accuracy is micro avg in classification report")