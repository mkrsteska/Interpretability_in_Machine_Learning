import json
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataLoader():

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
        le = preprocessing.LabelEncoder()
        self.df[self.class_column] = le.fit_transform(self.df[self.class_column])
        
    def split_target_labels(self): 
        self.y = self.df[self.class_column]        
        self.X = self.df.drop(self.class_column, axis=1)

        self.X = self.X.reset_index(drop=True)
        self.y = self.y.reset_index(drop=True)
       
    def standardize_numerical_features(self): 
        columns = list(self.X.select_dtypes(include = ['number']).columns)  

        standardize = StandardScaler()
        standardize.fit(self.X[columns], self.y)
        self.X[columns] = standardize.transform(self.X[columns])
        
        if self.separate_train_test_file == "True":
            standardize.fit(self.df_test[columns])
            self.df_test[columns] = standardize.transform(self.df_test[columns])
        
    def get_dummies(self):
        columns = list(self.X.select_dtypes(include = ['object']).columns)  
        self.X = pd.get_dummies(self.X, columns = columns)
        
        if self.separate_train_test_file == "True":
            self.df_test = pd.get_dummies(self.df_test, columns = columns)
            
    def train_test_split(self):
        if self.separate_train_test_file == "True":
            self.X_train = self.X
            self.y_train = self.y
            self.X_test = self.df_test
                    
        else: 
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=72)                    
    
    def sort_indexes(self):
        self.y_train.sort_values(inplace=True)
        self.X_train = self.X_train.reindex(self.y_train.index)
        
        if self.separate_train_test_file == "False":
            self.y_test.sort_values(inplace=True)
            self.X_test = self.X_test.reindex(self.y_test.index)
        
    def get_plot_sample(self, X_train, y_train):
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
        if (math.sqrt(number) - math.floor(math.sqrt(number)) != 0): 
            return False
        return True
    
    def get_closest_perfect_square(self, rows): 
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