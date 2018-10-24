#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 07:54:48 2018

@author: shaihe
"""

#import numpy as np
import pandas as pd

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
    data_id = data['claim_number']
    #data_x = data[data.columns[1:5]]
    #data_y = data['fraud']
    train_x, test_x, train_y, test_y = train_test_split(data[data.columns[1:5]], data['fraud'], test_size = 0.1)
    return data_id, train_x, train_y, test_x, test_y

def get_data2(file):
    # Create a set of dummy variables from the gender and genre variable
    df = pd.read_csv(file)
    df = df.dropna()
    df = df[df['annual_income'] > 0]
    data_id = df['claim_number']
    
    gender = pd.get_dummies(df['gender'])
    marital_status = pd.get_dummies(df['marital_status'])
    high_education_ind = pd.get_dummies(df['high_education_ind'])
    address_change_ind = pd.get_dummies(df['address_change_ind'])
    living_status = pd.get_dummies(df['living_status'])
    claim_day_of_week = pd.get_dummies(df['claim_day_of_week'])
    accident_site = pd.get_dummies(df['accident_site'])
    witness_present_ind = pd.get_dummies(df['witness_present_ind'])
    channel = pd.get_dummies(df['channel'])
    policy_report_filed_ind = pd.get_dummies(df['policy_report_filed_ind'])
    vehicle_category = pd.get_dummies(df['vehicle_category'])
    vehicle_color = pd.get_dummies(df['vehicle_color'])

    marital_status.columns=['no_married', 'married']
    high_education_ind.columns = ['no_high_education', 'high_education']
    address_change_ind.columns = ['no_address_change', 'address_change']
    witness_present_ind.columns = ['no_witness_present', 'witness_present']
    policy_report_filed_ind.columns = ['no_policy_report_filed', 'policy_report_filed']

    claim_day_of_week.drop(['Monday'], axis=1)
    vehicle_color.drop(['other'], axis=1)

    data = pd.concat([df[['claim_number', 'age_of_driver', 'safty_rating', 'annual_income', 'past_num_of_claims', 'liab_prct', 'past_num_of_claims']], gender['M'], marital_status['married'], high_education_ind['high_education'],address_change_ind['address_change'], living_status['Rent'], claim_day_of_week, accident_site[['Local','Highway']], channel[['Broker','Online']],policy_report_filed_ind['policy_report_filed'], vehicle_category[['Compact', 'Large']],vehicle_color], axis=1)
    
    #data_x = data[data.columns[1:5]]
    #data_y = data['fraud']
    train_x, test_x, train_y, test_y = train_test_split(df_new, data['fraud'], test_size = 0.1)
    return data_id, train_x, train_y, test_x, test_y



 
file_train = 'uconn_comp_2018_train.csv'
df = pd.read_csv(file_train)
df.head()
gender = pd.get_dummies(df['gender'])
marital_status = pd.get_dummies(df['marital_status'])
high_education_ind = pd.get_dummies(df['high_education_ind'])
address_change_ind = pd.get_dummies(df['address_change_ind'])
living_status = pd.get_dummies(df['living_status'])
claim_day_of_week = pd.get_dummies(df['claim_day_of_week'])
accident_site = pd.get_dummies(df['accident_site'])
witness_present_ind = pd.get_dummies(df['witness_present_ind'])
channel = pd.get_dummies(df['channel'])
policy_report_filed_ind = pd.get_dummies(df['policy_report_filed_ind'])
vehicle_category = pd.get_dummies(df['vehicle_category'])
vehicle_color = pd.get_dummies(df['vehicle_color'])

marital_status.columns=['no_married', 'married']
high_education_ind.columns = ['no_high_education', 'high_education']
address_change_ind.columns = ['no_address_change', 'address_change']
witness_present_ind.columns = ['no_witness_present', 'witness_present']
policy_report_filed_ind.columns = ['no_policy_report_filed', 'policy_report_filed']

claim_day_of_week.drop(['Monday'], axis=1)
vehicle_color.drop(['other'], axis=1)

df_new = pd.concat([df[['claim_number', 'age_of_driver', 'safty_rating', 'annual_income', 'past_num_of_claims', 'liab_prct', 'past_num_of_claims']], gender['M'], marital_status['married'], high_education_ind['high_education'],address_change_ind['address_change'], living_status['Rent'], claim_day_of_week, accident_site[['Local','Highway']], channel[['Broker','Online']],policy_report_filed_ind['policy_report_filed'], vehicle_category[['Compact', 'Large']],vehicle_color], axis=1)
df_new.head()





    


#using KNN classifier with default setting


def test_default(method, train_x, train_y, test_x, test_y):
    method_dict = {'KNN': KNeighborsClassifier(), 'CT': DecisionTreeClassifier(), 'NB':MultinomialNB(), 'LDA':LinearDiscriminantAnalysis(), 
                   'LR':LogisticRegression(), 'SVM':SVC(), 'NN': MLPClassifier()}
    
    t1 = time.process_time()#start time
    classifier = method_dict[method]
    classifier.fit(train_x, train_y)#fit model
    trainingtime = time.process_time() - t1
    score = classifier.score(test_x, test_y)
    predictiontime = time.process_time() - t1
    #return score and time
    return score, trainingtime, predictiontime


#using matplotlib to create a table to report different classifiers
def accuracytable(train_x, train_y, test_x, test_y):
    #get socre and time from different clasifiers
    method_dict = {'KNN': KNeighborsClassifier(), 'CT': DecisionTreeClassifier(), 'NB':MultinomialNB(), 'LDA':LinearDiscriminantAnalysis(), 
                   'LR':LogisticRegression(), 'SVM':SVC(), 'NN': MLPClassifier()}
    
    keep = []
    methodlist = list(method_dict.keys())
    for method in method_dict:
        score,tt,pt = test_default(method,train_x,train_y,test_x, test_y)
        keep.append([score,tt,pt])    
    
    #set columns names
    columns = ('Accuracy', 'Trainning Time', 'Prediction Time')
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
    
    
    
    
    