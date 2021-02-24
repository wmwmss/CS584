# this is the python implementation for the credit risk project

import os
import numpy as np
import pandas as pd
import re
from math import sqrt
import time
import string
import matplotlib
import matplotlib.pyplot as plt
import nltk
import sklearn

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.inspection import permutation_importance

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# set environment
os.chdir(r'C:\Users\wmwms\OneDrive - George Mason University\Academics\2020Fall\CS584\HW2')
os.getcwd()

# load data
train_data = pd.read_csv(r"data\1600106342_882043_train.csv", header = 0)#, nrows = 2000)

test_data = pd.read_csv(r"data\1600106342_8864183_test.csv", header = 0)#, nrows = 1000)
test_data.head
# explore
#train_data.head
#train_data.iloc[0]
#test_data.head
#test_data.iloc[0]
#train_data.dtypes
#train_data.describe()
#test_data.describe()

#train_data.isnull().sum()
#test_data.isnull().sum()

train_data.cov()
train_data.corr()

#%matplotlib inline
#train_data.boxplot()

#pd.plotting.parallel_coordinates(train_data,'credit')

# check y value distribution
train_data['credit'].value_counts()
# make train data balanced
temp_train_data = train_data[train_data['credit']==0]
temp_train_data_1 = train_data[train_data['credit']==1]
temp_train_data.shape
temp_train_data_1.shape

# balanced data has F1 score 0.64
#train_data =  temp_train_data.sample(n = 7841, replace = False).append(temp_train_data_1)
#train_data.shape

# increse 0 counts to 2x, F1 score 0.66
train_data =  temp_train_data.sample(n = 7841*2, replace = False).append(temp_train_data_1)
train_data.shape

# increse 0 counts to 3x, F1 score 0.65
#train_data =  temp_train_data.sample(n = 7841*3, replace = False).append(temp_train_data_1)
#train_data.shape

# data pre-processing
X_train = train_data.copy()
X_train = X_train.iloc[:,0:12]
Y_train = train_data['credit']
X_train.shape
Y_train.shape
X_test = test_data.copy()
X_test.shape

# convert string to numeric
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
X_train.groupby('F10')['F10'].nunique()
X_train["F10cat"] = lb_make.fit_transform(X_train["F10"])
X_train[["F10","F10cat"]].dtypes
X_train.groupby('F10cat')['F10cat'].nunique()
X_train['F10cat'].head
del X_train['F10']
X_train.head

lb_make = LabelEncoder()
X_train.groupby('F11')['F11'].nunique()
X_train["F11cat"] = lb_make.fit_transform(X_train["F11"])
X_train[["F11","F11cat"]].dtypes
X_train.groupby('F11cat')['F11cat'].nunique()
X_train['F11cat'].head
del X_train['F11']
del X_train['id']
X_train.head

lb_make = LabelEncoder()
X_test.groupby('F10')['F10'].nunique()
X_test["F10cat"] = lb_make.fit_transform(X_test["F10"])
X_test[["F10","F10cat"]].dtypes
X_test.groupby('F10cat')['F10cat'].nunique()
X_test['F10cat'].head
del X_test['F10']
X_test.head

lb_make = LabelEncoder()
X_test.groupby('F11')['F11'].nunique()
X_test["F11cat"] = lb_make.fit_transform(X_test["F11"])
X_test[["F11","F11cat"]].dtypes
X_test.groupby('F11cat')['F11cat'].nunique()
X_test['F11cat'].head
del X_test['F11']
del X_test['id']
X_test.head
new_test_data = X_test
Y_test = pd.DataFrame([0] * len(X_test.index))

# delete column(s)
del X_train['F3']
del X_test['F3']

new_test_data.shape
X_train.shape
Y_train.shape
X_test.shape
Y_test.shape

new_data_train = X_test
new_data_test = Y_test

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.33, shuffle = True)

# data standardization
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)


# data normalization
#X_train = preprocessing.normalize(X_train)
#X_test = preprocessing.normalize(X_test)
pd.DataFrame(X_train).cov()
pd.DataFrame(X_train).corr()

###################################################################################

# preliminary models
# Decision Trees
# F1: 0.58
'''
from sklearn import tree
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.8, random_state=1)

maxdepths = [2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]

trainAcc = np.zeros(len(maxdepths))
testAcc = np.zeros(len(maxdepths))

index = 0
for depth in maxdepths:
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(X_train, Y_train)
    Y_predTrain = clf.predict(X_train)
    Y_predTest = clf.predict(X_test)
    trainAcc[index] = accuracy_score(Y_train, Y_predTrain)
    testAcc[index] = accuracy_score(Y_test, Y_predTest)
    index += 1

plt.plot(maxdepths,trainAcc,'ro-',maxdepths,testAcc,'bv--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Max depth')
plt.ylabel('Accuracy')

'''

###################################################################################

'''
# KNN
# F1: 0.64
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
%matplotlib inline

numNeighbors = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
trainAcc = []
testAcc = []

for k in numNeighbors:
    clf = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    clf.fit(X_train, Y_train)
    Y_predTrain = clf.predict(X_train)
    Y_predTest = clf.predict(X_test)
    trainAcc.append(accuracy_score(Y_train, Y_predTrain))
    testAcc.append(accuracy_score(Y_test, Y_predTest))

plt.plot(numNeighbors, trainAcc, 'ro-', numNeighbors, testAcc,'bv--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')


'''

###################################################################################

# RandomForest
# F1: 0.64 with balanced data
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
numBaseClassifiers = 500
maxdepth = 10
trainAcc = []
testAcc = []
new_test_data.head


clf = ensemble.RandomForestClassifier(n_estimators=numBaseClassifiers)
clf.fit(X_train, Y_train)
Y_predTrain = clf.predict(X_train)
#Y_predTest = clf.predict(new_test_data)
#trainAcc.append(accuracy_score(Y_train, Y_predTrain))
#testAcc.append(accuracy_score(Y_test, Y_predTest))
#trainAcc
#testAcc

#Y_predTest

###################################################################################

# cross validation
# cv = KFold(n_splits = 10, random_state = 1, shuffle = True)
cv = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state = 1)
scores = cross_val_score(clf,X_train, Y_train, scoring = 'accuracy', cv = cv, n_jobs = -1)
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# F1 score
F1 = f1_score(Y_train,Y_predTrain, average = 'weighted')
print('F1 score: %.3f' % F1)

# feature importance
# found F3 to be of great importance yet its variance is large
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()

###################################################################################

# prediction
Y_predTest = clf.predict(new_test_data)

# output
sub = pd.DataFrame(Y_predTest)
#sub = sub[0:]
submission = open("submission.txt","w")
submission.write(sub.to_string(index = False))
submission.close()

###################################################################################
