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
import sys
sys.path.append('../_hepers_')
from genNewVals import *

warnings.filterwarnings('ignore')


def genSyntheticFillNulls(df):
    temp = []
    for i in df.columns:
        
        listWithNulls = df[i]
        numNulls = list(listWithNulls).count('?')
        nonNullColumn = list(filter(lambda val: val != '?', listWithNulls))
        nonNullColumn = list(map(float, nonNullColumn))

        hist, bins = np.histogram(nonNullColumn, bins=25)
        bin_midpoints = bins[:-1] + np.diff(bins)/2
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]
        values = np.random.rand(numNulls)
        value_bins = np.searchsorted(cdf, values)
        random_from_cdf = bin_midpoints[value_bins]

        count = 0
        for j in range(len(listWithNulls)):
            if listWithNulls[j] == '?':
                listWithNulls[j] = random_from_cdf[count]
                count += 1
        temp.append(listWithNulls)
    return pd.DataFrame(np.array(temp).T, columns=df.columns)

def removeEntriesWithNulls(df):
    temp = []
    for index, row in df.iterrows():
        if '?' not in row.values:
            temp.append(row)
    resultantDF = pd.DataFrame(np.array(temp), columns=df.columns)
    for i in resultantDF.columns:
        resultantDF[i] = resultantDF[i].astype(np.float)
    return resultantDF