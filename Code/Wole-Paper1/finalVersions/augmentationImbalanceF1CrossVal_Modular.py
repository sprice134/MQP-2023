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

def summarizeMeans(cumulativeList):
    meanList = []
    for i in cumulativeList:
        temp = []
        for j in range(len(i)):
            temp.append(np.mean(i[j]))
        meanList.append(temp)    
    return meanList


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

            
    df = pd.DataFrame(np.array(temp), columns=cols)
    df = df[cols]
    for i in df:
        df[i] = df[i].astype(np.int)
    return df, cols

def calculateWBC(percentY, typeOfBalance):
    leaveRAW = False
    if percentY == -1:
        leaveRAW = True
    random.seed(42)
    np.random.seed(42)
    df, cols = prepWBC()
    X = df[cols[1:-1]]
    y = df[cols[-1]]
    numY = math.ceil((len(y) - sum(y)) * percentY)
    percentWBC = sum(y) / len(y)
    if leaveRAW == False: #Only subsample if leaveRaw is false
        print('X: {}, Y: {}, Total: {}, SubY: {}'.format(len(y) - sum(y), sum(y), len(y), numY))
        subSample = pd.DataFrame(subsample(X, y, 20000, numY))
        X, y = subSample[subSample.columns[:-1]], subSample[subSample.columns[-1]]
    
    if typeOfBalance == 'SYN':
        numClassSamples = len(y) - sum(y)
        finalDataset = generateSupplementalData(X, y, numBenign = numClassSamples, numMalig = numClassSamples)
        X = finalDataset[finalDataset.columns[:-1]]
        y = finalDataset[finalDataset.columns[-1]]
    elif typeOfBalance == 'SMOTE':
        sm = SMOTE(random_state=42)
        X, y = sm.fit_resample(X, y)
    else:
        pass #No Balancing Required
    nb, log, svm, dt, vc = trainF1CrossValModels(X, y, 10)
    return nb, log, svm, dt, vc


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


def calculateWDBC(percentY, typeOfBalance):
    leaveRAW = False
    if percentY == -1:
        leaveRAW = True
    random.seed(42)
    np.random.seed(42)
    df, cols = prepWDBC()
    X = df[cols[:-1]]
    y = df[cols[-1]]
    numY = math.ceil((len(y) - sum(y)) * percentY)
    if leaveRAW == False: #Only subsample if leaveRaw is false
        print('X: {}, Y: {}, Total: {}, SubY: {}'.format(len(y) - sum(y), sum(y), len(y), numY))
        subSample = pd.DataFrame(subsample(X, y, 20000, numY))
        X, y = subSample[subSample.columns[:-1]], subSample[subSample.columns[-1]]
     
    if typeOfBalance == 'SYN':
        numClassSamples = int(len(y) - sum(y))
        print(numClassSamples)
        finalDataset = generateSupplementalData(X, y, numBenign = numClassSamples, numMalig = numClassSamples)
        X = finalDataset[finalDataset.columns[:-1]]
        y = finalDataset[finalDataset.columns[-1]]
    elif typeOfBalance == 'SMOTE':
        sm = SMOTE(random_state=42)
        X, y = sm.fit_resample(X, y)
    else:
        pass #No Balancing Required
    nb, log, svm, dt, vc = trainF1CrossValModels(X, y, 10)
    return nb, log, svm, dt, vc

def prepWDBC_stripped():
    cols = ['ID', 'Diagnosis', 'radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension',
        'radius_SE', 'texture_SE', 'perimeter_SE', 'area_SE', 'smoothness_SE', 'compactness_SE', 'concavity_SE', 'concave_points_SE', 'symmetry_SE', 'fractal_dimension_SE',
        'radius_MAX', 'texture_MAX', 'perimeter_MAX', 'area_MAX', 'smoothness_MAX', 'compactness_MAX', 'concavity_MAX', 'concave_points_MAX', 'symmetry_MAX', 'fractal_dimension_MAX']

    df = pd.read_csv('../../../Wisconsin_Database/wdbc.data', header=None)

    df.columns = cols
    df['Diagnosis'] = df['Diagnosis'].map(
                    {'M':1,'B':0})
    for i in df:
        df[i] = df[i].astype(np.float)
    
    
    colsInUse = ['ID', 'Diagnosis', 'texture', 'smoothness', 'symmetry', 'fractal_dimension', 
                'radius_SE', 'texture_SE', 'smoothness_SE', 'compactness_SE', 'concavity_SE', 'concave_points_SE', 'symmetry_SE', 'fractal_dimension_SE',
                'texture_MAX', 'perimeter_MAX', 'smoothness_MAX', 'compactness_MAX', 'concavity_MAX', 'concave_points_MAX', 'symmetry_MAX', 'fractal_dimension_MAX']

    df = df[colsInUse]

    colsV2 = colsInUse[2:]
    colsV2.append(colsInUse[1])
    df = df[colsV2]
    return df, colsV2

def calculateWDBC_stripped(percentY, typeOfBalance):
    leaveRAW = False
    if percentY == -1:
        leaveRAW = True
    random.seed(42)
    np.random.seed(42)
    df, cols = prepWDBC_stripped()
    X = df[cols[:-1]]
    y = df[cols[-1]]
    numY = math.ceil((len(y) - sum(y)) * percentY)
    if leaveRAW == False: #Only subsample if leaveRaw is false
        print('X: {}, Y: {}, Total: {}, SubY: {}'.format(len(y) - sum(y), sum(y), len(y), numY))
        subSample = pd.DataFrame(subsample(X, y, 20000, numY))
        X, y = subSample[subSample.columns[:-1]], subSample[subSample.columns[-1]]
     
    if typeOfBalance == 'SYN':
        numClassSamples = int(len(y) - sum(y))
        print(numClassSamples)
        finalDataset = generateSupplementalData(X, y, numBenign = numClassSamples, numMalig = numClassSamples)
        X = finalDataset[finalDataset.columns[:-1]]
        y = finalDataset[finalDataset.columns[-1]]
    elif typeOfBalance == 'SMOTE':
        sm = SMOTE(random_state=42)
        X, y = sm.fit_resample(X, y)
    else:
        pass #No Balancing Required
    nb, log, svm, dt, vc = trainF1CrossValModels(X, y, 10)
    return nb, log, svm, dt, vc

def prepSurgical():
    surgicalDF = pd.read_csv('Surgical-deepnet.csv')
    surgicalDF.rename(columns = {'complication':'Diagnosis'}, inplace = True)
    for i in surgicalDF:
        surgicalDF[i] = surgicalDF[i].astype(np.float)
    return surgicalDF, surgicalDF.columns


def calculateSurgical(percentY, typeOfBalance):
    leaveRAW = False
    if percentY == -1:
        leaveRAW = True
    random.seed(42)
    np.random.seed(42)
    df, cols = prepSurgical()
    X = df[cols[:-1]]
    y = df[cols[-1]]
    numY = int(math.ceil(percentY * 5000 / (1 - percentY))) #Extra care do to subsampling both due to computational time
    if leaveRAW == False: #Only subsample if leaveRaw is false
        print('X: {}, Y: {}, Total: {}, SubY: {}'.format(len(y) - sum(y), sum(y), len(y), numY))
        subSample = pd.DataFrame(subsample(X, y, 5000, numY)) #Extra care do to subsampling both due to computational time
        X, y = subSample[subSample.columns[:-1]], subSample[subSample.columns[-1]]
        print(X.shape)
    if typeOfBalance == 'SYN':
        numClassSamples = int(len(y) - sum(y))
        print(numClassSamples)
        finalDataset = generateSupplementalData(X, y, numBenign = numClassSamples, numMalig = numClassSamples)
        X = finalDataset[finalDataset.columns[:-1]]
        y = finalDataset[finalDataset.columns[-1]]
    elif typeOfBalance == 'SMOTE':
        sm = SMOTE(random_state=42)
        X, y = sm.fit_resample(X, y)
    else:
        pass #No Balancing Required
    nb, log, svm, dt, vc = trainF1CrossValModels(X, y, 10)
    return nb, log, svm, dt, vc



IMBALANCE_PARAMETERS = [.25, 0.2, 0.1, 0.05]
cumAugScores_surgical = []
cumSmoteScores_surgical = []
cumRawScores_surgical = []

for i in IMBALANCE_PARAMETERS:
    percentY = i
    print(percentY)
    augScores = calculateSurgical(percentY, 'SYN')
    cumAugScores_surgical.append(augScores)
    #augNBWBC, augLogWBC, augSVMWBC, auglDTWBC, augVoteWBC = calculateWBC(percentY, 'SYN')
    printOutputs(augScores[0], augScores[1], augScores[2], augScores[3], augScores[4])
    print('-'*30)

    smoteScores = calculateSurgical(percentY, 'SMOTE')
    cumSmoteScores_surgical.append(smoteScores)
    #smoteNBWBC, smoteLogWBC, smoteSVMWBC, smotelDTWBC, smoteVoteWBC = calculateWBC(percentY, 'SMOTE')
    printOutputs(smoteScores[0], smoteScores[1], smoteScores[2], smoteScores[3], smoteScores[4])
    print('-'*30)

    rawScores = calculateSurgical(percentY, 'RAW')
    cumRawScores_surgical.append(rawScores)
    #rawNBWBC, rawLogWBC, rawSVMWBC, rawlDTWBC, rawVoteWBC = calculateWBC(percentY, 'RAW')
    printOutputs(rawScores[0], rawScores[1], rawScores[2], rawScores[3], rawScores[4])
    print('-'*100)

meanAugScores_surgical= summarizeMeans(cumAugScores_surgical)
meanSmoteScores_surgical = summarizeMeans(cumSmoteScores_surgical)
meanRawScores_surgical = summarizeMeans(cumRawScores_surgical)

print('RAW DATA')
summarizedRaw_surgical = []
for i in meanRawScores_surgical:
    print('{} -> {:.04}'.format(i, np.mean(i)))
    summarizedRaw_surgical.append(np.mean(i))

print('SMOTE DATA')
summarizedSmote_surgical = []
for i in meanSmoteScores_surgical:
    print('{} -> {:.04}'.format(i, np.mean(i)))
    summarizedSmote_surgical.append(np.mean(i))

print("Synthetic DATA")
summarizedSyn_surgical = []
for i in meanAugScores_surgical:
    print('{} -> {:.04}'.format(i, np.mean(i)))
    summarizedSyn_surgical.append(np.mean(i))


surgical_x_vals = IMBALANCE_PARAMETERS
plt.plot(surgical_x_vals, summarizedRaw_surgical, label='RAW')
plt.plot(surgical_x_vals, summarizedSmote_surgical, label='SMOTE')
plt.plot(surgical_x_vals, summarizedSyn_surgical, label='SYN')
plt.legend(title='Dataset',loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
plt.savefig('SurgicalGraph.png')