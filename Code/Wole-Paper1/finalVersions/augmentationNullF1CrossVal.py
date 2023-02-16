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
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
import warnings #to remove the warnings
import random
import sys
sys.path.append('../../_hepers_')
from genNewVals import generateNewFeatureValMultiple
from subsampleData import subsample
from models import trainModels
from genFillNulls import *
warnings.filterwarnings('ignore')
from imblearn.over_sampling import SMOTE 
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer








def trainF1CrossValModels(X, y, numFolds):
    nb = GaussianNB()
    nbScores = cross_val_score(nb, X, y, cv=numFolds, scoring='f1')
    print('Completed Naive Bayes')
    logisticClassifier = LogisticRegression()
    logScores = cross_val_score(logisticClassifier, X, y, cv=numFolds, scoring='f1')
    print('Completed Logistic Regression')
    svclassifier = SVC(kernel='linear', max_iter = 5000000)
    svmScores = cross_val_score(svclassifier, X, y, cv=numFolds, scoring='f1')
    print('Completed SVM')
    clf = DecisionTreeClassifier(random_state = 42)
    dtScrores = cross_val_score(clf, X, y, cv=numFolds, scoring='f1')
    print('Completed Decision Tree')
    votingCl = VotingClassifier(
                estimators =    [('gnb', GaussianNB()),
                                ('lr',  LogisticRegression()),
                                ('svm', SVC(kernel='linear', max_iter = 5000000)),
                                ('dtc', DecisionTreeClassifier(random_state=42))], 
                voting='hard')
    voteScores = cross_val_score(votingCl, X, y, cv=numFolds, scoring='f1')
    print('Completed Voting Classification')
    return nbScores, logScores, svmScores, dtScrores, voteScores

def generateSupplementalData(X_train, y_train, numBenign = 0, numMalig = 0):
    trainingSet = X_train.copy()
    trainingSet['Diagnosis'] = y_train
    arr2D = trainingSet.to_numpy()
    columnIndex = -1
    sortedArr = arr2D[arr2D[:,columnIndex].argsort()]
    numZeros = list(sortedArr[:, columnIndex]).count(0)
    benignDF, malignantDF = pd.DataFrame(sortedArr[0:numZeros]), pd.DataFrame(sortedArr[numZeros:])
    benignDF.columns, malignantDF.columns = trainingSet.columns, trainingSet.columns
    numBenignNeeded = numBenign - benignDF.shape[0]
    numMaligNeeded = numMalig - malignantDF.shape[0]

    #Storing all Raw Benign Data to Later be Merged with Synthetic
    preExistingBenign = []
    for index, row in benignDF.iterrows():
        preExistingBenign.append(list(row))
    preExistingBenign = np.array(preExistingBenign)
    
    #Storing all Raw Malignant Data to Later be Merged with Synthetic
    preExistingMalig = []
    for index, row in malignantDF.iterrows():
        preExistingMalig.append(list(row))
    preExistingMalig = np.array(preExistingMalig)

    #Generating New Data
    benignPatients = []
    for col in benignDF.columns:
        benignPatients.append(generateNewFeatureValMultiple(benignDF, col, 25, numBenignNeeded))
    benignPatients = np.array(benignPatients).T
    malignangPatients = []
    for col in malignantDF.columns:
        malignangPatients.append(generateNewFeatureValMultiple(malignantDF, col, 25, numMaligNeeded))
    malignangPatients = np.array(malignangPatients).T

    #Combining Real Data and Synthetic Data, or Only using Real Data in the case of majority class
    if np.array(benignPatients).shape[0] > 0:
        benignPatients = np.vstack((preExistingBenign, np.array(benignPatients)))
    else:
        benignPatients = preExistingBenign
    if np.array(malignangPatients).shape[0] > 0:
        malignangPatients = np.vstack((preExistingMalig, np.array(malignangPatients)))
    else:
        benignPatients = preExistingBenign
    jointArray = np.vstack((benignPatients,malignangPatients))
    finalDataset = pd.DataFrame(jointArray, columns = benignDF.columns) 

    #Reporting Summary
    print('Final Dataset Composition: {}'.format(finalDataset.shape))
    print(' - Benign:')
    print('   -   New Benign Instances Created: {}'.format(numBenignNeeded))
    print('   -   Old Benign Instances Used: {}'.format(preExistingBenign.shape[0]))
    print(' - Malignant:')
    print('   -   New Malignant Instances Created: {}'.format(numMaligNeeded))
    print('   -   Old Malignant Instances Used: {}'.format(preExistingMalig.shape[0]))
    return finalDataset

def printOutputs(naiveBayesScore, LogScore, SVMScore, DTScore, VoteScore):
    print('NB Mean Scores: {}'.format(np.mean(naiveBayesScore)))
    print('LR Mean Scores: {}'.format(np.mean(LogScore)))
    print('SVM Mean Scores: {}'.format(np.mean(SVMScore)))
    print('DT Mean Scores: {}'.format(np.mean(DTScore)))
    print('VC Mean Scores: {}'.format(np.mean(VoteScore)))

def printDataComposition(X_train, y_train):
    trainingSet = X_train.copy()
    trainingSet['Diagnosis'] = y_train
    arr2D = trainingSet.to_numpy()
    columnIndex = -1
    sortedArr = arr2D[arr2D[:,columnIndex].argsort()]
    numZeros = list(sortedArr[:, columnIndex]).count(0)
    benignDF, malignantDF = pd.DataFrame(sortedArr[0:numZeros]), pd.DataFrame(sortedArr[numZeros:])
    benignDF.columns, malignantDF.columns = trainingSet.columns, trainingSet.columns

    #Reporting Summary
    print(' - Benign Instances: {}'.format(benignDF.shape[0]))
    print(' - Malignant Instances: {}'.format(malignantDF.shape[0]))

def genRandomReplaceIndex(numEntries, lastDataColumn):
    return randint(0, numEntries), randint(0, lastDataColumn)

def meanImputationByClass(X_train, y_train):
    trainingSet = X_train.copy()
    trainingSet['Diagnosis'] = y_train
    arr2D = trainingSet.to_numpy()
    columnIndex = -1
    sortedArr = arr2D[arr2D[:,columnIndex].argsort()]
    numZeros = list(sortedArr[:, columnIndex]).count(0)
    benignDF, malignantDF = pd.DataFrame(sortedArr[0:numZeros]), pd.DataFrame(sortedArr[numZeros:])
    benignDF.columns, malignantDF.columns = trainingSet.columns, trainingSet.columns
    print('Benign: {}, Benign w/o Nulls: {}'.format(benignDF.shape, benignDF.dropna().shape))
    print('Malig: {}, Malig w/o Nulls: {}'.format(malignantDF.shape, malignantDF.dropna().shape))
    bening_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    bening_imputer = bening_imputer.fit(benignDF)
    benignDF = bening_imputer.transform(benignDF)

    malig_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    malig_imputer = malig_imputer.fit(malignantDF)
    malignantDF = malig_imputer.transform(malignantDF)
    jointArray = np.vstack((benignDF,malignantDF))
    return jointArray

def removeNulls(inputList):
    outputList = []
    for i in inputList:
        if str(i) != 'nan':
            outputList.append(i)
    return outputList


def syntheticImputaitonByClass(X_train, y_train):
    trainingSet = X_train.copy()
    trainingSet['Diagnosis'] = y_train
    arr2D = trainingSet.to_numpy()
    columnIndex = -1
    sortedArr = arr2D[arr2D[:,columnIndex].argsort()]
    numZeros = list(sortedArr[:, columnIndex]).count(0)
    benignDF, malignantDF = pd.DataFrame(sortedArr[0:numZeros]), pd.DataFrame(sortedArr[numZeros:])
    benignDF.columns, malignantDF.columns = trainingSet.columns, trainingSet.columns
    print('Benign: {}, Benign w/o Nulls: {}'.format(benignDF.shape, benignDF.dropna().shape))
    print('Malig: {}, Malig w/o Nulls: {}'.format(malignantDF.shape, malignantDF.dropna().shape))
    for i in benignDF:
        #print(i)
        numValsNeeded = list(map(str, list(benignDF[i]))).count('nan')
        nums = removeNulls(benignDF[i])
        #--------
        hist, bins = np.histogram(nums, bins=25)
        bin_midpoints = bins[:-1] + np.diff(bins)/2
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]
        values = np.random.rand(numValsNeeded + 1)
        value_bins = np.searchsorted(cdf, values)
        random_from_cdf = np.floor(np.array(bin_midpoints[value_bins]))
        #print(random_from_cdf)
        count = 0
        columnTempVals = list(benignDF[i])
        #print(columnTempVals)
        for j in range(len(columnTempVals)):
            if str(columnTempVals[j]) == 'nan':
                columnTempVals[j] = random_from_cdf[count]
                count += 1
        benignDF[i] = columnTempVals
    for i in malignantDF:
        #print(i)
        numValsNeeded = list(map(str, list(malignantDF[i]))).count('nan')
        nums = removeNulls(malignantDF[i])
        #--------
        hist, bins = np.histogram(nums, bins=25)
        bin_midpoints = bins[:-1] + np.diff(bins)/2
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]
        values = np.random.rand(numValsNeeded + 1)
        value_bins = np.searchsorted(cdf, values)
        random_from_cdf = np.floor(np.array(bin_midpoints[value_bins]))
        #print(random_from_cdf)
        count = 0
        columnTempVals = list(malignantDF[i])
        #print(columnTempVals)
        for j in range(len(columnTempVals)):
            if str(columnTempVals[j]) == 'nan':
                columnTempVals[j] = random_from_cdf[count]
                count += 1
        malignantDF[i] = columnTempVals
    jointArray = np.vstack((np.array(benignDF),np.array(malignantDF)))
    return jointArray








def removeAtRandom(df, cols, percentRemoved):
    print(df.shape)
    print(df.dropna().shape)
    numEntriesNeedingNulls = math.floor(df.shape[0] * percentRemoved)
    numEntriesWithNulls = df.shape[0] - df.dropna().shape[0]
    array = np.array(df)
    for i in range(numEntriesNeedingNulls - numEntriesWithNulls):
        x_index, y_index = genRandomReplaceIndex(array.shape[0], array.shape[1] - 1)
        array[x_index][y_index] = np.nan
    df = pd.DataFrame(array, columns = cols)
    return df, cols

def prepWBC():
    raw_df = pd.read_csv('../../../Wisconsin_Database/breast-cancer-wisconsin.data', header=None)
    cols = ['ID', 'Thickness', 'SizeUniformity', 'ShapeUniformity', 'Adhesion', 'Size', 'BareNuclei', 'Bland Chromatin', 'Nucleoli', 'Mitoses', 'Diagnosis']
    raw_df.columns = cols
    raw_df['Diagnosis'] = raw_df['Diagnosis'].map(
                    {2:0,4:1})
    temp = []
    for index, row in raw_df.iterrows():
        if '?' not in row.values:
            temp.append(row)
        else:
            temp.append(row.replace('?', np.nan))

            
    df = pd.DataFrame(np.array(temp), columns=cols)
    df = df[cols]
    return df, cols

def prepWDBC():
    cols = ['ID', 'Diagnosis', 'radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension',
        'radius_SE', 'texture_SE', 'perimeter_SE', 'area_SE', 'smoothness_SE', 'compactness_SE', 'concavity_SE', 'concave_points_SE', 'symmetry_SE', 'fractal_dimension_SE',
        'radius_MAX', 'texture_MAX', 'perimeter_MAX', 'area_MAX', 'smoothness_MAX', 'compactness_MAX', 'concavity_MAX', 'concave_points_MAX', 'symmetry_MAX', 'fractal_dimension_MAX']

    df = pd.read_csv('../../../Wisconsin_Database/wdbc.data', header=None)

    df.columns = cols
    df['Diagnosis'] = df['Diagnosis'].map(
                    {'M':1,'B':0})
    for i in df:
        df[i] = df[i].astype(np.float)

    colsV2 = cols[2:]
    colsV2.append(cols[1])
    df = df[colsV2]
    return df, colsV2

def prepCervical():
    '''cervicalCols = ['Age', 'Number of sexual partners', 'First sexual intercourse',
                    'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
                    'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
                    'IUD (years)', 'STDs (number)', 'STDs:condylomatosis',
                    'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis',
                    'STDs:syphilis', 'STDs:pelvic inflammatory disease',
                    'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:HIV',
                    'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
                    'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis',
                    'Dx:Cancer', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology',
                    'Diagnosis']'''
    cervical_df = pd.read_csv('risk_factors_cervical_cancer.csv')
    cervical_df.rename(columns = {'Biopsy':'Diagnosis'}, inplace = True)
    cols = cervical_df.columns
    cervical_df.replace('?', np.nan)
    
    temp = []
    for index, row in cervical_df.iterrows():
        if '?' not in row.values:
            temp.append(row)
        else:
            temp.append(row.replace('?', np.nan))
    df = pd.DataFrame(np.array(temp), columns=cols)
    df = df[cols]

    for i in df:
        df[i] = df[i].astype(np.float)
    return df, cols

iterativeMeanSurgical = []
iterativeRemovalSurgical = []
iterativeSynSurgical = []

for removalValue in ([0.1, 0.33, 0.5, 0.66, 0.9]):
    print('NEW ORDERING: {}'.format(removalValue))
    print('-'*50)

    random.seed(42)
    np.random.seed(42)
    df = pd.read_csv('Surgical-deepnet.csv')
    df.rename(columns = {'complication':'Diagnosis'}, inplace = True)
    cols = df.columns
    for i in df:
        df[i] = df[i].astype(np.float)
    subSample = df.sample(5000)
    df, cols = removeAtRandom(subSample, cols, removalValue)
    print(df.dropna().shape)
    bening_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    bening_imputer = bening_imputer.fit(df)
    df = pd.DataFrame(bening_imputer.transform(df), columns=cols)
    print('New DF Shape: {}'.format(df.shape))
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]
    for i in finalDataset:
        finalDataset[i] = finalDataset[i].astype(np.int)
    X = finalDataset[finalDataset.columns[:-1]]
    y = finalDataset[finalDataset.columns[-1]]

    temp = trainF1CrossValModels(X, y, 10)
    iterativeRemovalSurgical.append(temp)
    removeNBSurgical, removeLogSurgical, removeSVMSurgical, removeDTSurgical, removeVoteSurgical = temp
    printOutputs(removeNBSurgical, removeLogSurgical, removeSVMSurgical, removeDTSurgical, removeVoteSurgical)
    

    print('-' * 25)


    random.seed(42)
    np.random.seed(42)
    df = pd.read_csv('Surgical-deepnet.csv')
    df.rename(columns = {'complication':'Diagnosis'}, inplace = True)
    cols = df.columns
    for i in df:
        df[i] = df[i].astype(np.float)
    print(df)
    subSample = df.sample(5000)
    df, cols = removeAtRandom(subSample, cols, removalValue)
    print(df.dropna().shape)
    bening_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    bening_imputer = bening_imputer.fit(df)
    df = pd.DataFrame(bening_imputer.transform(df), columns=cols)
    print('New DF Shape: {}'.format(df.shape))
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]
    for i in finalDataset:
        finalDataset[i] = finalDataset[i].astype(np.int)
    X = finalDataset[finalDataset.columns[:-1]]
    y = finalDataset[finalDataset.columns[-1]]
    temp = trainF1CrossValModels(X, y, 10)
    iterativeMeanSurgical.append(temp)
    meanNBSurgical, meanLogSurgical, meanSVMSurgical, meanDTSurgical, meanVoteSurgical = trainF1CrossValModels(X, y, 10)
    printOutputs(meanNBSurgical, meanLogSurgical, meanSVMSurgical, meanDTSurgical, meanVoteSurgical)


    print('-'*25)
    

    random.seed(42)
    np.random.seed(42)
    df = pd.read_csv('Surgical-deepnet.csv')
    df.rename(columns = {'complication':'Diagnosis'}, inplace = True)
    cols = df.columns
    for i in df:
        df[i] = df[i].astype(np.float)

    subSample = df.sample(5000)
    df, cols = removeAtRandom(subSample, cols, removalValue)
    print(df.dropna().shape)
    X = df[cols[1:-1]]
    y = df[cols[-1]]

    finalDataset = pd.DataFrame(syntheticImputaitonByClass(X, y))
    finalDataset.columns = cols[1:]
    for i in finalDataset:
        finalDataset[i] = finalDataset[i].astype(np.int)
    X = finalDataset[finalDataset.columns[:-1]]
    y = finalDataset[finalDataset.columns[-1]]

    temp = trainF1CrossValModels(X, y, 10)
    iterativeMeanSurgical.append(temp) 
    synNBSurgical, synLogSurgical, synSVMSurgical, synDTSurgical, synVoteSurgical = trainF1CrossValModels(X, y, 10)
    printOutputs(synNBSurgical, synLogSurgical, synSVMSurgical, synDTSurgical, synVoteSurgical)

