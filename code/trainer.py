import json
import pandas as pd
import numpy as np
import seaborn
import math
import random
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from ast import literal_eval as make_tuple

class Trainer():
    """
    Class to train the models and plot the results.
    
    Parameters
    ----------
    data_loader : DataLoader, required 
        DataLoader object. Contains the dataset. 
    
    Attributes
    ----------
    reference_model : sklearn.base.BaseEstimator
        Reference model. Performs the role of a teacher in the student model.

    models : list 
        List of student models.
        
    labels : list 
        Name of student models. Used for plotting the results.
        
    model_predictions : list
        Contains list of predictions from each model.
        
    reference_model_y_test_predictions : list 
        Predictions from `reference_model`
        
    X_train_X_test : pandas.DataFrame
        The augmented train dataset. Consists of `self.data_loader.X_train` and `self.data_loader.X_test`. Contains one row for each instance, and one column for each feature. 
    
    y_train_y_test : numpy.ndarray
        The outputs for the augmented train dataset. Contains the target label for every instance in `X_train_X_test`. The outputs for the instances in `self.data_loader.X_test` are obtained with the reference model. 
        
    matrix1 : numpy.ndarray
        Contains the number of matching predictions for each pair of models on `self.data_loader.X_train`. 
    
    matrix2 : numpy.ndarray
        Contains the number of matching predictions for each pair of models on `X_train_X_test`. 
        
    blended : list 
        List of shades of colors according to the probability that the corresponding instance will be in the target class.
    
    to_plot : list 
        List of predictions colored according to the probability that the corresponding instance will be in the target class.
    """
        
    def __init__(self, data_loader):
        with open("constants.json") as f:
            self.CONSTANTS = json.load(f)    
        
        self.data_loader = data_loader
        
        reference_model_dict = self.CONSTANTS["reference_model"].popitem()
        self.reference_model = globals()[reference_model_dict[0]]()
        for attribute_name, attribute_value in reference_model_dict[1].items():
            attribute_value = self.get_attribute_value(attribute_value)
            setattr(self.reference_model, attribute_name, attribute_value)           
                     
        self.models = [self.reference_model]
             
        for model, values in self.CONSTANTS["models"].items(): 
            instance = globals()[model]()
            for attribute_name, attribute_value in values.items():
                attribute_value = self.get_attribute_value(attribute_value)
                setattr(instance, attribute_name, attribute_value)
            self.models.append(instance)    
                                        
        self.labels = [x.__class__.__name__ for x in self.models]
        
        print("Plotting the predictions for a sample of X_train set")
        
        self.model_predictions = []     
        self.predict(self.data_loader.X_train, self.data_loader.y_train)        
        self.matrix1 = self.create_confusion_matrix()
        
        print("Plotting the predictions for a sample of X_train + more labeled points given by the reference model")
        self.reference_model.fit(self.data_loader.X_train, self.data_loader.y_train)

        self.reference_model_y_test_predictions = self.reference_model.predict(self.data_loader.X_test)
        
        self.X_train_X_test = pd.concat([self.data_loader.X_train, self.data_loader.X_test], axis=0)
        self.y_train_y_test = np.concatenate((self.data_loader.y_train.to_numpy(), self.reference_model_y_test_predictions), axis=0)
        
        self.model_predictions = []      
        self.predict(self.X_train_X_test, self.y_train_y_test)
        self.matrix2 = self.create_confusion_matrix()
        
        print("Plotting heatmaps")
        self.plot_confusion_matrices()
             
    def get_attribute_value(self, attribute_value):
        """
        Converts and returns the value of `attribute_value` in the required type: bool, None, int, float, string, tuple.

        Parameters
        ----------
        attribute_value : string, required 
            The value of the attribute in type string, as read from the json file.  

        Returns
        -------
        attribute_value : bool / None / int / float / tuple / string    
        """
        if (attribute_value == "True"):
            return True
        elif (attribute_value == "False"):
            return False
        elif (attribute_value == "None"):
            return None
        elif (self.is_number(attribute_value)):
            if attribute_value.isdecimal():
                return int(attribute_value)
            else:
                return float(attribute_value)
        else: 
            try:
                return make_tuple(attribute_value)
            except:   
                return attribute_value
    
    def is_number(self, attribute_value):
        """
        Returns `True` if `attribute_value` is a number, `False` otherwise.

        Parameters
        ----------
        attribute_value : string, required
            The value of the attribute in type string, as read from the json file.  

        Returns
        -------
        is_number: bool
            Whether `attribute_value` is number.
        """
        try:
            float(attribute_value)
            return True
        except:
            return False
    
    def predict(self, X_train, y_train):
        """
        Predict the target labels with each model, store the predictions in `self.model_predictions` and plot a sample of the predictions.

        Parameters
        ----------
        X_train : pandas.DataFrame, required
            The input train dataset. Contains one row for each instance, and one column for each feature.

        y_train : pandas.Series, required
            The output train dataset. Contains the target label for every instance in `X_train`.

        """
        self.plot_sample = self.data_loader.get_plot_sample(X_train, y_train)
        self.number = int(math.sqrt(self.data_loader.square_number))

        for self.model in self.models:
            self.model.fit(X_train, y_train)
            self.model_predictions.append(self.model.predict(X_train))
            
            self.blend_sample_predictions()
            self.plot_sample_predictions() 
               
    def blend_sample_predictions(self):
        """
        Colours the predictions in shades according to the probability that the corresponding instance will be in the target class.
        """
        self.blended = []
        probabilities_sample = self.model.predict_proba(self.plot_sample)
        
        colors = self.get_colors()

        for p in probabilities_sample:
            r, g, b = 0, 0, 0
            for i in range(len(colors)):
                r, g, b = r+p[i]*colors[i][0], g+p[i]*colors[i][1], b+p[i]*colors[i][2]
            self.blended.append([r, g, b])    
            
    def get_colors(self):
        """
        Returns list of colors in rgb format, one for each class. 

        Returns
        -------
        colors : list 
            Returns list with as many elements as number of classes.
        """
        colors = [[1,0,0], [0,1,0], [0,0,1], [1,0.5, 0], [0, 1, 0.5], [0.5, 0, 1]]
        return colors[:self.data_loader.num_classes]

    def plot_sample_predictions(self):         
        """
        Plot the predictions for a sample of the dataset. There is one square for each data instance in the sample, colored according to the probability that the instance is in a given target class.
        """
        self.to_plot = []
        k = 0
        for i in range(self.number):
            row = []
            for j in range(self.number):
                row.append(self.blended[k])
                k+=1
            self.to_plot.append(row)

        plt.title(self.model.__class__.__name__)
        plt.imshow(self.to_plot);
        plt.show()           
   
    def create_confusion_matrix(self):
        """
        Returns matrix. Each entry is sum of matching predictions for each pair of models. 

        Returns
        -------
        matrix : numpy.ndarray
            Returns confusion matrix.
        """
        matrix = np.zeros((len(self.models), len(self.models)))
        for i, pred1 in enumerate(self.model_predictions):
            for j, pred2 in enumerate(self.model_predictions):
                matrix[i, j] = sum([1 for x, y in zip(pred1, pred2) if x == y])
        return matrix
    
    def plot_confusion_matrices(self):
        """
        Plots heatmaps using the normalized values from `self.matrix1` and `self.matrix2`. Additionally, plots a heatmap with the relative improvement in matching predictions for each pair of models when using the augmented dataset.  
        """
        
        plt.figure(figsize=(16, 12))
        plt.subplot(211)
        plt.title("Percentage of matching predictions for each pair of models for the initial test set")
        seaborn.heatmap(self.matrix1 / self.matrix1.max() * 100, xticklabels = self.labels, yticklabels = self.labels, cmap='Blues', annot=True, fmt='g')
        
        plt.subplot(212)
        plt.title("Percentage of matching predictions for each pair of models when using the augmented dataset")
        seaborn.heatmap(self.matrix2 / self.matrix2.max() * 100, xticklabels = self.labels, yticklabels = self.labels, cmap='Blues', annot=True, fmt='g')
        
        plt.figure(figsize=(10, 6))
        plt.title('Relative improvement in matching predictions')
        seaborn.heatmap(self.matrix2 / self.matrix2.max() * 100 - self.matrix1 / self.matrix1.max() * 100, xticklabels = self.labels, yticklabels = self.labels, cmap='Blues', annot=True, fmt='g')