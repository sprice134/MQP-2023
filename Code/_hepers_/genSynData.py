import matplotlib
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix
import warnings #to remove the warnings
import random
from genNewVals import *
warnings.filterwarnings('ignore')




def generateSyntheticDataset(X_train, y_train, identical = True, numBenign = 0, numMalig = 0):
    trainingSet = X_train.copy()
    trainingSet['Diagnosis'] = y_train
    arr2D = trainingSet.to_numpy()
    columnIndex = -1
    sortedArr = arr2D[arr2D[:,columnIndex].argsort()]
    numZeros = list(sortedArr[:, columnIndex]).count(0)
    benignDF, malignantDF = pd.DataFrame(sortedArr[0:numZeros]), pd.DataFrame(sortedArr[numZeros:])
    #print('benignDF: {}'.format(benignDF.shape))
    #print('malignantDF: {}'.format(malignantDF.shape))
    benignDF.columns = trainingSet.columns
    malignantDF.columns = trainingSet.columns
    benignPatients = []
    if identical:
        for col in benignDF.columns:
            benignPatients.append(generateNewFeatureValMultiple(benignDF, col, 25, benignDF.shape[0]))
        beningPatients = np.array(benignPatients).T
        malignangPatients = []
        for col in malignantDF.columns:
            malignangPatients.append(generateNewFeatureValMultiple(malignantDF, col, 25, malignantDF.shape[0]))
        malignangPatients = np.array(malignangPatients).T
    else:
        for col in benignDF.columns:
            benignPatients.append(generateNewFeatureValMultiple(benignDF, col, 25, numBenign))
        beningPatients = np.array(benignPatients).T
        malignangPatients = []
        for col in malignantDF.columns:
            malignangPatients.append(generateNewFeatureValMultiple(malignantDF, col, 25, numMalig))
        malignangPatients = np.array(malignangPatients).T
    #print('benignPatients: {}'.format(np.array(beningPatients).shape))
    #print('malignantPatients: {}'.format(np.array(malignangPatients).shape))
    jointArray = np.vstack((beningPatients,malignangPatients))
    finalDataset = pd.DataFrame(jointArray, columns = benignDF.columns)   
    return finalDataset


