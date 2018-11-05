import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import csv

from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn import preprocessing, cross_validation, svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

import pickle

def test_predictions(df, actual, predicted):
    conditions = [
    (df[predicted] > 0)]
    choices = [1]
    df['Simulated Trade'] = np.select(conditions, choices, default=0)
    df['Simulated Return'] = df['Simulated Trade'] * df[actual]

    conditions2 = [
    (df['Simulated Return'] > 0)]
    choices2 = [1]
    df['Simulated Return Wins'] = np.select(conditions2, choices2, default=0)

    conditions3 = [
    (df['Simulated Return'] < 0)]
    df['Simulated Return Losses'] = np.select(conditions3, choices2, default=0)

    wins = df['Simulated Return Wins'].sum()
    losses = df['Simulated Return Losses'].sum()
    total = wins + losses

    winpercentage = round(((wins/total) * 100), 2)
    print('(Long only) Simulated ML Win Percentage: ' + str(winpercentage) + '% (' + str(wins) + '/' + str(total) + ')')

    df = df.drop(columns=['Simulated Return Wins', 'Simulated Return Losses'])

    return df

def data_cleaner(df, trigger, trig_val):
    conditions = [
    (df[trigger] == trig_val)]
    choices = [1]
    df['Cleaner Row'] = np.select(conditions, choices, default=0)
    df = df[df['Cleaner Row'] != 0]
    df = df.drop(columns=['Cleaner Row'])
    return df

def data_trainer(df, algo, y, x1, x2=None, x3=None, x4=None, x5=None, x6=None, x7=None, x8=None, x9=None, x10=None):
    checklist = [x1]
    if x2 != None:
        checklist.append(x2)
    if x3 != None:
        checklist.append(x3)
    if x4 != None:
        checklist.append(x4)
    if x5 != None:
        checklist.append(x5)
    if x6 != None:
        checklist.append(x6)
    if x7 != None:
        checklist.append(x7)
    if x8 != None:
        checklist.append(x8)
    if x9 != None:
        checklist.append(x9)
    if x10 != None:
        checklist.append(x10)
    X = df[checklist]
    y = df[y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #training
    if algo == 'DecisionTreeRegressor':
        regressor = DecisionTreeRegressor()
    elif algo == 'DecisionTreeClassifier':
        regressor = DecisionTreeClassifier()
    elif algo == 'SVC':
        regressor = SVC(kernel='rbf', probability=True)
    elif algo == 'GaussianNB':
        regressor = GaussianNB()
    elif algo == 'RandomForestClassifier':
        regressor = RandomForestClassifier()
    elif algo == 'KNeighbors':
        regressor = KNeighborsClassifier()
    elif algo == 'MLP':
        regressor = MLPClassifier()

    regressor.fit(X_train, y_train)

    #prediction
    y_pred = regressor.predict(X_test)

    prediction = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) 
    print(prediction)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
    print()

    '''
    pickle.dump(regressor, open(targetFile, 'wb'))
    print('Model saved: ', targetFile)
    print()

    loaded_model = pickle.load(open(targetFile, 'rb'))
    result = loaded_model.score(X_test, y_test)
    result = round((result*100), 2)
    print('Confidence: ', result)
    '''

    return prediction
