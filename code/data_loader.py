import json
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataLoader():
    """
        Class to load the data.

        Attributes
        ----------
        CONSTANTS : dict
            Dictionary containing the variables from the json configuration file
        
        separate_train_test_file : string
            Whether the train and test data are in separate files. 

        class_column : string
            Name of the target class. 

        absolute_path_train : string 
            Absolute path to the train data. Used only if `separate_train_test_file` is `True`.
        
        absolute_path_test : string 
            Absolute path to the test data. Used only if `separate_train_test_file` is `True`.

        absolute_path : string 
            Absolute path to the data. Used only if `separate_train_test_file` is `False`. 

        df : pandas.DataFrame
            Contains the train dataset, ff `separate_train_test_file` is `True`. Otherwise, contains the whole dataset. 
        
        df_test : pandas.DataFrame 
            Contains the test dataset. Used only if `separate_train_test_file` is `True`.
        
        num_classes : int
            Number of unique classes.   
            
        X : pandas.DataFrame 
            Contains one row for each instance, and one column for each feature.
        
        y : pandas.Series 
            Contains the target label for every instance in `X`.
            
        X_train : pandas.DataFrame
            The input train dataset. Contains one row for each instance, and one column for each feature.
        
        y_train : pandas.Series 
            The output train dataset. Contains the target label for every instance in `X_train`.

        X_test : pandas.DataFrame
            The input test dataset. Contains one row for each instance, and one column for each feature.

        y_test : pandas.Series
            The output test dataset. Contains the target label for every instance in `X_test`.
            
        square_number : int 
            The largest square number equal or smaller than the number of rows in the dataset. Used for plotting.
            Square number is an integer that is the square of an integer.   
        """

    def __init__(self):
        
        with open("constants.json") as f:
            self.CONSTANTS = json.load(f)
                    
        self.separate_train_test_file = self.CONSTANTS["separate_train_test_file"]
        self.class_column = self.CONSTANTS["class_column"]
                
        if self.separate_train_test_file == "True":
            self.absolute_path_train = self.CONSTANTS["absolute_path_train"]
            self.absolute_path_test = self.CONSTANTS["absolute_path_test"]   
            
            self.df = pd.read_csv(self.absolute_path_train)
            self.df_test = pd.read_csv(self.absolute_path_test) 
                                   
        else:
            self.absolute_path = self.CONSTANTS["absolute_path"]
            self.df = pd.read_csv(self.absolute_path)
             
        self.num_classes = self.df[self.class_column].nunique()

        self.encode_target_labels()

        self.split_target_labels()

        self.standardize_numerical_features()

        self.get_dummies()

        self.train_test_split()

        self.sort_indexes() 
                
    def encode_target_labels(self):
        """
        Encodes target labels with values between 0 and num_classes-1.
        """
        le = preprocessing.LabelEncoder()
        self.df[self.class_column] = le.fit_transform(self.df[self.class_column])
        
    def split_target_labels(self): 
        """
        Splits the  target labels from the rest of the dataset.               
        """
        self.y = self.df[self.class_column]        
        self.X = self.df.drop(self.class_column, axis=1)

        self.X = self.X.reset_index(drop=True)
               
    def standardize_numerical_features(self): 
        """
        Standardizes numerical features by removing the mean and scaling to unit variance.
        """
        columns = list(self.X.select_dtypes(include = ['number']).columns)  

        standardize = StandardScaler()
        standardize.fit(self.X[columns], self.y)
        self.X[columns] = standardize.transform(self.X[columns])
        
        if self.separate_train_test_file == "True":
            standardize.fit(self.df_test[columns])
            self.df_test[columns] = standardize.transform(self.df_test[columns])
        
    def get_dummies(self):
        """
        Converts categorical variables into dummy/indicator variables.
        """
        columns = list(self.X.select_dtypes(include = ['object']).columns)  
        self.X = pd.get_dummies(self.X, columns = columns)
        
        if self.separate_train_test_file == "True":
            self.df_test = pd.get_dummies(self.df_test, columns = columns)
            
    def train_test_split(self):
        """
        Splits the matrices `self.X` and `self.y` into random train and test subsets.
        """ 
        if self.separate_train_test_file == "True":
            self.X_train = self.X
            self.y_train = self.y.copy()
            self.X_test = self.df_test
                    
        else: 
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=72)  
               
    def sort_indexes(self):
        """
        Sorts the instances of `X_train` and 'X_test` by the target label in `y_train` and `y_test` accordingly, in ascending order.         
        """ 
        self.y_train.sort_values(inplace=True)
        self.X_train = self.X_train.reindex(self.y_train.index)
            
        if self.separate_train_test_file == "False":            
            self.y_test.sort_values(inplace=True)
            self.X_test = self.X_test.reindex(self.y_test.index)
        
    def get_plot_sample(self, X_train, y_train):
        """
        Returns a sample of the train dataset given as a paremeter. All classes in the sample are represented with same proportion as in the original dataset. 
        
        Parameters
        ----------
        X_train : pandas.DataFrame, required
            The complete input train dataset. Contains one row for each instance, and one column for each feature.
            
        y_train : pandas.DataFrame, required
            The complete output train dataset. Contains the target label for every instance in `X_train`.
            
        Returns
        -------
        df_tmp : pandas.DataFrame 
            Sample of the input train dataset. Contains one row for each instance, and one column for each feature.
        
        """
        df_sample = X_train.copy()
        df_sample[self.class_column] = y_train.copy()

        df_sample_rows = df_sample.shape[0]

        self.get_closest_perfect_square(df_sample_rows)
        
        df_tmp = pd.DataFrame()
        for i in range(self.num_classes-1):
            class_i = df_sample[df_sample[self.class_column]==i].shape[0]/float(df_sample_rows)
            class_i_sample = int(self.square_number*class_i)
            df_class_i = df_sample[df_sample[self.class_column]==i].sample(class_i_sample)
            df_tmp = pd.concat([df_tmp, df_class_i])
        
        df_tmp_num_rows = df_tmp.shape[0]
        class_nth_sample = self.square_number - df_tmp_num_rows
        
        df_class_nth = df_sample[df_sample[self.class_column]==self.num_classes-1].sample(class_nth_sample)
        df_tmp = pd.concat([df_tmp, df_class_nth])

        df_tmp = df_tmp.drop(self.class_column, axis=1)

        return df_tmp
     
    def is_perfect(self, number):
        """
        Returns `True` if `number` is a square number, `False` otherwise.
        
        Parameters
        ----------
        number : int, required
            The number to check.
        
        Returns
        -------
        is_perfect : bool
            Whether `number` is a square number.
        """
        is_perfect = True
        if (math.sqrt(number) - math.floor(math.sqrt(number)) != 0): 
            is_perfect = False
        return is_perfect
    
    def get_closest_perfect_square(self, rows):
        """
        Find the largest square number equal or smaller than the number of rows in the dataset and assign it to `self.square_number`
        
        Parameters
        ----------
        rows : int, required 
            Number of instances in the dataset.       
        """
        if (self.is_perfect(rows)):  
            self.square_number = rows 

        else: 
            number = rows - 1
            while (True): 
                if (self.is_perfect(number)):  
                    self.square_number = number  
                    break
                else: 
                    number -= 1