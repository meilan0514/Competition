#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 07:54:48 2018

@author: shaihe
"""

#import numpy as np
import pandas as pd
import numpy as np

import matplotlib
matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

import time
#import sklearn.model_selection as model_selection
#import kaggle

'''
#need to clean up the data
#eg: annual income negative and veichle age contains na: ages has 200 
'''


#load data, adapted from TA post
def get_data(FILE):
    df = pd.read_csv(FILE)
    df = df.dropna()
    data = df[df['annual_income'] > 0]
    data =data[data['fraud'] > -1]
    data_id = data['claim_number']
    #data_x = data[data.columns[1:5]]
    #data_y = data['fraud']
    train_x, test_x, train_y, test_y = train_test_split(data[data.columns[1:5]], data['fraud'], test_size = 0.1)
    return data_id, train_x, train_y, test_x, test_y


#using KNN classifier with default setting


def test_default(method, train_x, train_y, test_x, test_y):
    method_dict = {'KNN': KNeighborsClassifier(), 'CT': DecisionTreeClassifier(), 
                   #'NB':MultinomialNB(), 
                   'LDA':LinearDiscriminantAnalysis(), 
                   'LR':LogisticRegression(), 
                   #'SVM':SVC(), 
                   'NN': MLPClassifier()}
    
    t1 = time.process_time()#start time
    classifier = method_dict[method]
    classifier.fit(train_x, train_y)#fit model
    trainingtime = time.process_time() - t1
    score = classifier.score(test_x, test_y)
    prob = classifier.predict_proba(test_x)
    roc_score = roc_auc_score(np.asarray(test_y), np.asarray(prob[:,1]))
    predictiontime = time.process_time() - t1
    #return score and time
    return score, trainingtime, predictiontime, prob, roc_score


#using matplotlib to create a table to report different classifiers
def accuracytable(train_x, train_y, test_x, test_y):
    #get socre and time from different clasifiers
    method_dict = {'KNN': KNeighborsClassifier(), 'CT': DecisionTreeClassifier(), 
                   #'NB':MultinomialNB(), 
                   'LDA':LinearDiscriminantAnalysis(), 
                   'LR':LogisticRegression(), 
                   #'SVM':SVC(), 
                   'NN': MLPClassifier()}
    
    keep = []
    methodlist = list(method_dict.keys())
    for method in method_dict:
        score,tt,pt,_, roc = test_default(method,train_x,train_y,test_x, test_y)
        keep.append([score,tt,pt,roc])    
    
    #set columns names
    columns = ('Accuracy', 'Trainning Time', 'Prediction Time', 'roc_score')
    #set row names
    rows = methodlist
    data = keep
    fig, ax = plt.subplots()
    #turn off axis
    ax.axis('off')
    ax.axis('tight')
    #build table
    ax.table(cellText=data,rowLabels=rows,colLabels=columns,loc='center')
    plt.savefig('accuracy.png')
    plt.close()
    
    

#Find optimal hyperparameter by cross-validation
#Here I choose RF classifier to opimalzation
def tuning(method,train,seed):
    method_dict = {'KNN': KNeighborsClassifier(), 'CT': DecisionTreeClassifier(), 'NB':MultinomialNB(), 'LDA':LinearDiscriminantAnalysis(), 
                   'LR':LogisticRegression(), 'SVM':SVC(), 'NN': MLPClassifier()}
    p_grid = {"n_estimators": [10,25,50,100],
    #to reduce calculation time only tune one hyperprameter
          }
    classfier = GridSearchCV(estimator=method_dict[method], param_grid=p_grid)
    classfier.fit(train[0],train[1])
    result = classfier.cv_results_
    score = result['mean_test_score']#getting socre result
    para = result['params']#all parameter
    return score, para 

##make line graph to report differnet accuracy   
#def hplot(method, train,seed):
#    #get y value
#    score, __ = tuning(train, seed)
#    x= [10,25,50,100]
#    fig, ax = plt.subplots()
#    #turn on axis since it turns off when creating the table
#    ax.axis('on')
#    plt.plot(x, score)
#    plt.show()
    



#def best(seed):
#    #best settings, set n_estimbators = 1000
#    best = RandomForestClassifier(random_state = seed, n_estimators = 1000)
#    #fit the model
#    best.fit(train[0],train[1])
#    testerror = best.score(test[0],test[1])
#    #make prediction
#    pred = best.predict(kaggledata)
#    #make a csv file to upload
#    kaggle.kaggleize(pred,"/Users/shaihe/Desktop/pred_cv.csv")
#    #print adjusted hyperparameters 
#    print("Best Model: Random Forest,n_estimators = 1000", "accuracy on test set is ", testerror)
#

if __name__ == '__main__':
    
    
    File_1 = 'test_uconn_comp_2018_train_1.csv'
    #File_2 = 'tees_uconn_comp_2018_test.csv'
    #main function, read data
    
    train_id, train_x, train_y, test_x, test_y= get_data(File_1)
    #set seed to 5
    #seed = 5
    
    
    accuracytable(train_x, train_y, test_x,test_y)#output the table
    #hplot(train,seed)#output the line graph
    #print(tuningdefault(train, test, seed)) #print out the re-sub result
    #best(seed) #create best model
    
    