Semester Project - spring 2020 

This repository contains Python implementation of the Surrogate Model on various datasets. 

# Overview #

Surrogate model is an interpretable machine learning model, that is trained to approximate the behaviour of a well-performing black-box model. The main goal is to build more interpretable models, without decreasing performance. 

In this repository, the surrogate models are created with the following steps.

1. Select a dataset and split it on training and testing parts
2. Train a black box model on the training dataset
3. Obtain the predictions of the black box model on the testing dataset
4. Create a new training dataset by merging the original training dataset and the testing dataset with the labels provided by the black-box model
5. Train the surrogate model on the new training dataset.

Several different surrogate models are trained and compared based on three criteria:
- How well they approximate the black-box model
- The final performance
- The improvement in performance when compared to their non-surrogate version

Finally, the procedure is repeated on several different datasets to explore the impact of the data on the performance of the surrogate models.

# Models #

The current black-box model is a Neural Network with one hidden layer.

The following models are used as surrogate models:

- Gaussian Naive Bayes  
- K Nearest Neighbours
- Random Fores
- Logistic Regression

# Datasets #

The models are trained on the following datasets

- Titanic https://www.kaggle.com/c/titanic/data
- Banknote_Authentication https://www.kaggle.com/shantanuss/banknote-authentication-uci
- Breast_Cancer https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
- Heart_Disease https://www.kaggle.com/ronitf/heart-disease-uci
- Mushrooms https://www.kaggle.com/uciml/mushroom-classification
- Rain_Australia https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
- Telco_Customer_Churn https://www.kaggle.com/blastchar/telco-customer-churn

# Current results #

# Future work #

Next steps:
- tuning of hyperparameters
- implement more complex black-box models 
- try out more datasets 
