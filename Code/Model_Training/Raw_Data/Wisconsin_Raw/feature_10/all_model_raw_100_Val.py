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
warnings.filterwarnings('ignore')


raw_df = pd.read_csv('../../../../../Wisconsin_Database/breast-cancer-wisconsin.data', header=None)
cols = ['ID', 'Thickness', 'SizeUniformity', 'ShapeUniformity', 'Adhesion', 'Size', 'BareNuclei', 'Bland Chromatin', 'Nucleoli', 'Mitoses', 'Diagnosis']
raw_df.columns = cols
raw_df['Diagnosis'] = raw_df['Diagnosis'].map(
                   {2:0,4:1})

temp = []
for index, row in raw_df.iterrows():
    if '?' not in row.values:
        temp.append(row)
        
df = pd.DataFrame(np.array(temp), columns=cols)
df['BareNuclei'] = df['BareNuclei'].astype(np.int)
print(df.shape)

knnScores = []
naiveBaysScores = []
logisticRegressionScores = []
svmRegressionScores = []
decisionTreeRegressionScores = []

for i in range(100):

    X = df[cols[1:-1]]
    y = df[cols[-1]]
    y=y.astype('int')
    SEED = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=SEED+i)
    if i == 0:
        print(X_train.shape)
    scaler = StandardScaler()
    scaler.fit(X_train)
    col_names=df.columns[1:-1]
    scaled_df = pd.DataFrame(X_train, columns=col_names)
    
    ''' KNN '''

    error = []
    score_vals = []
    for j in range(1, 100):
        knn = KNeighborsRegressor(n_neighbors=j)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        mae = mean_absolute_error(y_test, pred_i)
        error.append(mae)
        score_vals.append(knn.score(X_test, y_test))
    bestIndex = score_vals.index(max(score_vals)) + 1  #Adds one because it starts with 1 neighbor not 0
    print(bestIndex)
    regressor = KNeighborsRegressor(n_neighbors=bestIndex)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    y_pred = y_pred == 1
    #print('Iteration {}: '.format(i))
    performanceAccuracy = sum(y_pred == y_test)/len(y_test)
    knnScores.append(performanceAccuracy)

    '''Naive Bayes'''

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = (nb.predict(X_test))
    performanceAccuracy = sum(y_pred == y_test)/len(y_test)
    naiveBaysScores.append(performanceAccuracy)

    '''Logistic Regression'''

    modelLogistic = LogisticRegression()
    modelLogistic.fit(X_train,y_train)
    y_pred = modelLogistic.predict(X_test)
    performanceAccuracy = sum(y_pred == y_test)/len(y_test)
    logisticRegressionScores.append(performanceAccuracy)

    '''SVM'''

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    performanceAccuracy = sum(y_pred == y_test)/len(y_test)
    svmRegressionScores.append(performanceAccuracy)
    
    '''Decision Tree'''

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    performanceAccuracy = sum(y_pred == y_test)/len(y_test)
    decisionTreeRegressionScores.append(performanceAccuracy)

    print('Iteration: {}'.format(i))
results = pd.DataFrame()
results['KNN'] = knnScores
results['NB'] = naiveBaysScores
results['LR'] = logisticRegressionScores
results['SVM'] = svmRegressionScores
results['DT'] = decisionTreeRegressionScores
#results.to_csv('allModel90-10Cross100.csv')

print('KNN - Min: {}, Max: {}, Avg: {}'.format(min(knnScores), max(knnScores), sum(knnScores)/len(knnScores)))
print('   Best Seed: {}'.format(42 + knnScores.index(max(knnScores))))
print('NB  - Min: {}, Max: {}, Avg: {}'.format(min(naiveBaysScores), max(naiveBaysScores), sum(naiveBaysScores)/len(naiveBaysScores)))
print('   Best Seed: {}'.format(42 + naiveBaysScores.index(max(naiveBaysScores))))
print('LR  - Min: {}, Max: {}, Avg: {}'.format(min(logisticRegressionScores), max(logisticRegressionScores), sum(logisticRegressionScores)/len(logisticRegressionScores)))
print('   Best Seed: {}'.format(42 + logisticRegressionScores.index(max(logisticRegressionScores))))
print('SVM - Min: {}, Max: {}, Avg: {}'.format(min(svmRegressionScores), max(svmRegressionScores), sum(svmRegressionScores)/len(svmRegressionScores)))
print('   Best Seed: {}'.format(42 + svmRegressionScores.index(max(svmRegressionScores))))
print('DT  - Min: {}, Max: {}, Avg: {}'.format(min(decisionTreeRegressionScores), max(decisionTreeRegressionScores), sum(decisionTreeRegressionScores)/len(decisionTreeRegressionScores)))
print('   Best Seed: {}'.format(42 + decisionTreeRegressionScores.index(max(decisionTreeRegressionScores))))