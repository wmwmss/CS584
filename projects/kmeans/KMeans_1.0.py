# this is the python implementation of K-Means Clustering part1

import os
import numpy as np
import pandas as pd
import scipy
from scipy.spatial import distance
import re
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import f1_score
from math import sqrt
import time
import string
#from nltk.stem import PorterStemmer
#from nltk.corpus import stopwords

# set environment
os.chdir(r'C:\Users\wmwms\OneDrive - George Mason University\Academics\2020Fall\CS584\HW3\part1')
os.getcwd()

# load data
data = pd.read_csv(r"test.txt", sep = " ", header = None)#, header = 0)#, nrows = 2000)
#data.head
data[0].shape

#####################################################################################################
# test
'''
df = pd.DataFrame({
    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
})
input = df.values
# normalize
sc = StandardScaler()
sc.fit(input)
input = sc.transform(input)

index = np.random.choice(len(input), 3, replace=False)
index
input[2,:]
centroids = input[index, :]
centroids
t = distance.cdist(input, centroids, 'euclidean')
t
# calculate euclidean distance between each point and each centroid, get the nearest centroids
dist = np.argmin(t,axis=1)
dist
# for max rounds calculate new centroids
k = 3
for i in range(100):
    # calculate euclidean distance between each point and each centroid, get the nearest centroids
    centroids = np.vstack([input[dist==j,:].mean(axis=0) for j in range(k)])
    temp = np.argmin(distance.cdist(input, centroids, 'euclidean'),axis=1)
    # calculate error
    error = np.linalg.norm(temp - dist)
    print("error: %d" % error)
    if np.array_equal(dist,temp):break
    dist = temp
dist
'''

# normalize
data_array = data.values
sc = StandardScaler()
sc.fit(data_array)
data_array = sc.transform(data_array)

#####################################################################################################

#rounds = 0

def kmeans(input, k, maxrounds):
    # randomly choose k numbers as starting index
    index = np.random.choice(len(input), k, replace=False)
    # choose initial centroids based on the index
    centroids = input[index, :]
    # calculate min euclidean distance between each point and each centroid
    # find the nearest centroid based on the min distance, return its cluster
    cluster = np.argmin(distance.cdist(input, centroids, 'euclidean'),axis=1)
    # within max rounds calculate new centroids
    error = 0
    for i in range(maxrounds):
        # calculate new centroids
        centroids = np.vstack([input[cluster==j,:].mean(axis=0) for j in range(k)])
        # calculate new clusters
        temp = np.argmin(distance.cdist(input, centroids, 'euclidean'),axis=1)

        #rounds = rounds + 1
        # calculate error
        error += np.linalg.norm(temp - cluster)
        print("error: %d" % error)
        # if the error does not increase and centroids do not change, return
        if np.array_equal(cluster,temp):break
        # otherwise assign new clusters
        cluster = temp
    return cluster

# test data
clusters = kmeans(data_array, 3, 100)
# assert clusters size
assert len(data_array) == len(clusters)

# denormalize
data_array = sc.inverse_transform(data_array)
#data_array
plt.figure(figsize = (15, 10))
plt.scatter(data_array[:,0], data_array[:,1], c = clusters)
plt.show()

plt.figure(figsize = (15, 10))
plt.plot(range(6), [5,10,15,17,20,20])
plt.show()

#####################################################################################################

# output
sub = pd.DataFrame(clusters)
#sub
sub = sub[0:]
submission = open("submission.txt","w")
submission.write(sub.to_string(index = False))
submission.close()

#####################################################################################################

