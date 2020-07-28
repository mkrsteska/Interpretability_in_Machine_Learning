import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import random

from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

class SyntheticData():
    """
    Class to create synthetic data and show an example of surrogate learning models.
    
    Attributes
    ----------
    models : list 
        List of models.
           
    points_1 : tuple list 
        List of tuples. These ordered pairs are the synthetically generated data points. 
        
    clusters : numpy.ndarray
        The points are assigned to clusters. 
    
    X : list
        The input train dataset.
            
    y : list
        The output train dataset.
    """
    
    def __init__(self):
        self.models = [MLPClassifier(), GaussianNB(), KNeighborsClassifier(), RandomForestClassifier(), LogisticRegression()]

        self.set_distribution()        
        self.generate_points()
        
        line_1 = np.linspace(-5, 5, 41)
        self.points_1 = self.cartesian(line_1)
        
        line_2 = np.linspace(-5, 4.75, 14)
        points_2 = self.cartesian(line_2)
                
        self.get_X_y()
        
        print("Classifiers trained on the original data")
        self.predict(self.X, self.y)
        
        print("Classifiers trained on the original data and the hard labels from the Neural Network")
        points_2_predictions = MLPClassifier().fit(self.X, self.y).predict(points_2)        
        self.predict(self.X + points_2, self.y + list(points_2_predictions))
        
    def set_distribution(self):
        """
        Set the distribution of the synthetic data. 
        """
        clusters = []
        for i in range(4):
            ro = random.uniform(0.2, 0.8)
            sigma1, sigma2 = random.uniform(0.5, 2), random.uniform(0.5, 2)
            mu1, mu2 = random.uniform(-3, 3), random.uniform(-3, 3)
            cov_matrix = [[sigma1*sigma1, ro*sigma1*sigma2], [ro*sigma1*sigma2, sigma2*sigma2]]
            clusters.append(np.random.multivariate_normal([mu1, mu2], cov_matrix, random.randint(30, 40)))
        self.clusters = np.array(clusters)
        
    def generate_points(self):
        """
        Generates synthetic points. 
        """    
        for cluster in self.clusters:
            points = [(point[0], point[1]) for point in cluster]
            x, y = zip(*points)
            plt.scatter(x, y)
        plt.title("Generated synthetic points")
        plt.show()           
        
    def cartesian(self, line):
        """
        Returns cartesian product from the dots on the line.
        
        Parameters
        ----------
        line : numpy.ndarray, required
        """
        return [[x, y] for x in line for y in line] 
    
    def get_X_y(self):
        """
        Assigns each point in a cluster and assigns a target label to it. 
        """
        self.X = []
        self.y = []
        for i, cluster in enumerate(self.clusters):
            self.X = self.X + [list(point) for point in cluster]
            self.y = self.y + [i for point in cluster]
                           
    def predict(self, X, y):
        """
        Predict the target labels with each model, store the predictions in `self.predictions_proba` and blend the predictions and plot them.

        Parameters
        ----------
        X : list, required
            The input points.

        y : list, required
            The output points.

        """
        for self.model in self.models:
            self.model.fit(X, y)
            self.predictions_proba = self.model.predict_proba(self.points_1)
            
            self.blend_predictions()
            self.plot_predictions()   
    
    def blend_predictions(self):
        """
        Colours the predictions in shades according to the probability that the corresponding instance will be in the target class.
        """
        self.blended = []
        colors=[[1, 0, 0], [0, 0, 1], [0, 1, 0], [0.7, 0.5, 0]]

        for p in self.predictions_proba:
            r, g, b = 0, 0, 0
            for i in range(len(colors)):
                r, g, b = r+p[i]*colors[i][0], g+p[i]*colors[i][1], b+p[i]*colors[i][2]
            self.blended.append([r, g, b])  
                
    def plot_predictions(self):
        """
        Plot the predictions for a sample of the dataset. There is one square for each data instance in the sample, colored according to the probability that the instance is in a given target class.
        """
        plt.title(self.model.__class__.__name__)
        plt.imshow(np.reshape(self.blended, (41, 41, 3)));
        plt.show()   