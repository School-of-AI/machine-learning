
dataset = read.csv('Data.csv')
# dataset = dataset[, 2:3] # slecting specific dataset to create our model

# Splitting the dataset into Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
# SplitRatio is ratio of training set
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# we will not scale Country and Purchased category since they are not numeric as they are factor
#training_set[, 2:3] = scale(training_set[, 2:3])
#test_set[, 2:3] = scale(test_set[, 2:3])