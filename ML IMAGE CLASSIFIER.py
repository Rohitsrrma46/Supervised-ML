#Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importing Loadmat from scipy library for loading Mat file dataset
from scipy.io import loadmat
#Loading the dataset using lodmat
dataset = loadmat("C:/Users/Rohit/Downloads/ML/mnist-original") 
#Creating Feature Matrix
X = dataset["data"].T  #In dataset Dictionary "data" key value is selected.
#also ".T" represents the Transpose of "X".
#IE "Rows and Colums are Interchanged.
y = dataset["label"].T #In dataset Dictionary "label" key value is selected and again Transpose.
# Transpose is done to Convert the Feature Rows into Feature Columns.


any_no= X[45854] #X[Row] 
any_no_image = any_no.reshape(28,28)#Reshaping the 784 colums of "X" in 28*28 Pixels,

plt.imshow(any_no_image)# Represents the 28*28 pixel image 
plt.show

from sklearn.tree import DecisionTreeClassifier# Importing the DecisionTreeClassifier 

dtf = DecisionTreeClassifier(max_depth=15)
dtf.fit(X,y)   
dtf.score(X,y)#Calculates the Accuracy of the model

dtf.predict(X[[45854],0:784])# Predicts the value for given Row of"X".
