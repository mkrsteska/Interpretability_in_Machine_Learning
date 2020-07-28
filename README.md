# Interpretability in Machine Learning #
This repository contains the code for the semester project done by Marija Krsteska, under the supervision of Professor Michalis Vlachos from University of Lausanne, Faculty of Business and Economics.

**The general area of this project is the topic of interpretability in Machine Learning. The idea is to evaluate how more understandable models fare against more complex ones. For this purpose, a Teacher - Student framework is developed and multiple algorithms are evaluated in the role of a student model. The framework is tested on several datasets.**

## Structure of the project ##
The project is structured as follows: 

**code/** 
  - ***Main Notebook.ipynb***  </br>  *Notebook that cointains detailed explanations on the project structure and usage.* 
  - synthetic_data.py  </br> *File used to provide a low-dimensional example of surrogate learning models that perform a type of mimic learning.*
  - constants.json  </br> *Configuration file that needs to be filled in by the user, in order to use the Teacher-Student framework.*
  - data_loader.py  </br> *File used to load the data.*
  - trainer.py  </br> *File used to train, plot and compare the models.*</br>
  - **preprocessing/**  </br> *Preprocessing steps for all datasets.*
    * Titanic.ipynb  </br>
    * Banknote_Authentication.ipynb  </br>
    * Breast_Cancer.ipynb  </br>
    * Heart_Disease.ipynb  </br>
    * Mushrooms.ipynb  </br>
    * Rain_Australia.ipynb  </br>
    * Telco_Customer_Churn.ipynb 
   
**data/**  </br> 
  - clean_data/  </br> *Folder that contains the processed data.*
  - raw_data/ </br> Folder *that contains the raw downloaded data.*

## Overview ##

Student model, or surrogate model, is an interpretable machine learning model, that is trained to mimic the behaviour of a well-performing black-box teacher model. The main goal is to build more interpretable models, without decreasing performance. 

In this repository, the student models are created with the following steps:

- Select a dataset and split it on training and testing set
- Train a teacher model on the training dataset, manually labelled data, as in supervised learning
- Obtain the predictions of the teacher model on the testing dataset
- Create a new training dataset, union dataset, by merging the original training dataset and the testing dataset with the hard labels provided by the teacher model
- Train the student model on the union dataset

Several different student models are trained and compared based on three criteria:
- How well they approximate the black-box model on the original test dataset
- How well they approximate the black-box model on the union dataset
- The relative improvement in approximation when compared to their non-surrogate version

Finally, the procedure is repeated on several different datasets to explore the impact of the data on the performance of the surrogate models.

## Models ##

The current teacher model is a Neural Network with one hidden layer.

The following models are used as student models:

- Gaussian Naive Bayes  
- K Nearest Neighbours
- Random Fores
- Logistic Regression

## Datasets ##

The models are trained on the following datasets
- [Titanic](https://www.kaggle.com/c/titanic/data)
- [Banknote Authentication](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
- [Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- [Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- [Mushrooms](https://archive.ics.uci.edu/ml/datasets/Mushroom)
- [Rain Australia](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)
- [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

## Results ##

From the results, it can be seen that for different datasets, different teacher-student combinations perform better or worse. This indicates that this approach is not as general, meaning that not any student can learn from any teacher.

## Future work ##
- Implement a more complex black-box teacher model 
- Test more teacher-student combinations
- Test more datasets
- Try conditional teacher-student learning [1]

## References ##
[1] Meng, Zhong, et al. "Conditional teacher-student learning." ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019.
</br>[2] Abbasi, Sajjad, et al. "Modeling Teacher-Student Techniques in Deep Neural Networks for Knowledge Distillation." 2020 International Conference on Machine Vision and Image Processing (MVIP). IEEE, 2020.
</br>[3] Radosavovic, Ilija, et al. "Data distillation: Towards omni-supervised learning." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
</br>[4] Interpretable Recommender Systems: Algorithms, Metrics and Visualization Techniques, Prof. Michalis Vlachos
</br>[5] Cho, Jang Hyun, and Bharath Hariharan. "On the efficacy of knowledge distillation." Proceedings of the IEEE International Conference on Computer Vision. 2019.
