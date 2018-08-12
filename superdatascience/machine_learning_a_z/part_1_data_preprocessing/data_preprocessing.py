# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import (Imputer, LabelEncoder, OneHotEncoder,
                                   StandardScaler)
from sklearn.cross_validation import train_test_split #use "model_selection", cross_validation is depcrecated

# Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding categorical data
label_encoder_x = LabelEncoder()
x[:, 0] = label_encoder_x.fit_transform(x[:, 0])

# Creating different columns based on different Countries
one_hot_encoder = OneHotEncoder(categorical_features=[0])
x = one_hot_encoder.fit_transform(x).toarray()

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_learn = train_test_split(x, y, test_size=0.2, random_state = 0)


# Feature Scaling -
"""
https://stackoverflow.com/questions/26225344/why-feature-scaling
https://en.wikipedia.org/wiki/Feature_scaling
"""
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

