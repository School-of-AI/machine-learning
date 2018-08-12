# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.cross_validation import train_test_split #use "model_selection", cross_validation is depcrecated

# Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_learn = train_test_split(x, y, test_size=0.2, random_state = 0)


# Feature Scaling -
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""
