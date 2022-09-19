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


def subsample(X_train, y_train, numBenign, numMalig):
    trainingSet = X_train.copy()
    trainingSet['Diagnosis'] = y_train
    arr2D = trainingSet.to_numpy()
    columnIndex = -1
    sortedArr = arr2D[arr2D[:,columnIndex].argsort()]
    numZeros = list(sortedArr[:, columnIndex]).count(0)
    benignDF = pd.DataFrame(sortedArr[0:min(numZeros, numBenign)])
    malignantDF =  pd.DataFrame(sortedArr[numZeros:numZeros + min(numMalig, len(sortedArr) - numZeros)])
    result = pd.concat([benignDF, malignantDF])
    result.columns = benignDF.columns
    for j in result:
        result[j] = result[j].astype(np.int)
    return result