# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:30:54 2019

@author: PeIgnacz
"""
#Data processing
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Data.csv')
#x = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, 3].values
x = pd.DataFrame(dataset.iloc[:, :-1].values)
y = pd.DataFrame(dataset.iloc[:, 3].values)

#Missing Data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
#imputer.fit(x[:, 1:3])
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(x.iloc[:, 1:3])
x.iloc[:, 1:3] = imputer.transform(x.iloc[:, 1:3])

#Encoding categorical data
#independent variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x.iloc[:,0] = labelencoder_x.fit_transform(x.iloc[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()
#dependent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Training/Test split
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)




