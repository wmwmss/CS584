# this is the python implementation of recommender systems for HW4
# raw data with decision trees 1.27, knn 1.13, RandomForest 1.31
# add more dimentions and features
# add directorID with decision trees 1.35, knn 1.10, RandomForest 1.31
# add genres dt 1.33

#collaborative filtering
# user row movie col
# knn user A
# k = 30, 1.20
# k = 50, 1.08
# k = 70, 1.02
# k = 90, 0.98
# k = 120, 0.95
# k = 150, 0.94
# k = 180, 0.92
# k = 250, 0.90
# k = 350, 0.88

# matrix factorization
# > 0.7, 1.09

import os
#import nltk
import numpy as np
import pandas as pd
import scipy
from scipy.spatial import distance
import re
%matplotlib inline
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import f1_score
from math import sqrt
import time
import string
#from nltk.stem import PorterStemmer
#from nltk.corpus import stopwords

# set environment
os.chdir(r'C:\Users\wmwms\OneDrive - George Mason University\Academics\2020Fall\CS584\HW4\data')
os.getcwd()

# load data
train = pd.read_csv(r"additional_files\train.dat", sep = ' ', header = 0)#, header = 0)#, nrows = 2000)
test = pd.read_csv(r"test.dat", sep = ' ', header = 0)#, header = 0)#, nrows = 2000)


movie_actors = pd.read_csv(r"additional_files\movie_actors.dat", sep = '\s+', encoding = 'unicode_escape', header = 0,usecols=[0,1], names=['movieID', 'actorID'])#, 'actorName', 'ranking'])
#movie_actors.head
movie_directors = pd.read_csv(r"additional_files\movie_directors.dat", sep = '\s+', encoding = 'unicode_escape', header = 0,usecols=[0,1], names=['movieID', 'directorID'])
#movie_directors.head
duplicate = movie_directors[movie_directors.duplicated()]
len(duplicate)


#movie_tags = pd.read_csv(r"additional_files\movie_tags.dat", sep = '\s+', header = 0)
#movie_tags.head
#tags = pd.read_csv(r"additional_files\tags.dat", sep = '\s+', header = 0)
#tags.head
#user_taggedmovies = pd.read_csv(r"additional_files\user_taggedmovies.dat", sep = '\s+', header = 0)
#user_taggedmovies.head
#data.head
#train.shape
labels = train.columns
#train.head

#test.shape
#test.head

X_train = train[['userID', 'movieID']]
Y_train = train[['rating']]

# pivot
#train
train = pd.pivot_table(train, index = ['userID'], columns = ['movieID'], values = ['rating'])
#train
# replace missing data with mean
train.fillna(0, inplace = True)


pd.DataFrame(train)
#train.loc[75]
#train.index == 75
#row_index =
#train.loc[train.index == 75]

train_movieIDs = []
train.columns.shape[0]
for c in range(train.columns.shape[0]):
    train_movieIDs.append(train.columns[c][1])
#train_movieIDs

#userIDs = []
train_userIDs = []
train.columns.shape[0]
for c in range(train.columns.shape[0]):
    train_userIDs.append(train.columns[c][0])





# encoding Y
lab_enc = preprocessing.LabelEncoder()
Y_train = lab_enc.fit_transform(Y_train)


#Y_train
#X_train.shape


# normalize
#data_array = data.values
#data_array = preprocessing.normalize(data_array, norm = 'l2')


#nor = Normalizer()
#nor.fit(data_array)
#data_array = nor.transform(data_array)
#sc = StandardScaler()
#sc.fit(data_array)
#data_array = sc.transform(data_array)

# sparse matrix
#data_array = csr_matrix(data_array)
#plt.spy(train)
#plt.title("sparse matrix")

'''
# add movie_genres
movie_genres = pd.read_csv(r"additional_files\movie_genres.dat", sep = '\s+', header = 0)
movie_genres.groupby('genre')['genre'].nunique().count()
movie_genres['genre'].isna().sum()

lab_enc1 = preprocessing.LabelEncoder()
movie_genres['genre'] = lab_enc1.fit_transform(movie_genres['genre'])

df_new = pd.crosstab(movie_genres['movieID'],movie_genres['genre']).rename_axis(None,axis=1)#.add_prefix('type@')
#df_new

X_train = X_train.merge(df_new, on = 'movieID', how='left')
test = test.merge(df_new, on = 'movieID', how='left')

X_train
test
'''


# sparse matrix
train_csr = csr_matrix(train.values)
plt.spy(train_csr)
plt.title("sparse matrix")

'''
X_train = csr_matrix(X_train)
plt.spy(X_train)
plt.title("sparse matrix")

test = csr_matrix(test)
plt.spy(test)
plt.title("sparse matrix")
'''



# SVD
from numpy import array
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=200)
svd.fit(train_csr)
svd_train_csv = svd.transform(train_csr)
svd_train_csv
svd.explained_variance_ratio_.sum()
svd.singular_values_

import warnings
warnings.filterwarnings("ignore", category = RuntimeWarning)
corr = np.corrcoef(svd_train_csv)
corr

movie_list = list(labels)
'''
######################################################################
    userID = test.iloc[0][0]
    # find userID in train
    train_record = train.loc[train.index == userID]
    train_record
    # find neighbors in train_csr
    neighbors = corr[userID]
    neighbors
    neighbors_index = list(neighbors[((neighbors<1.0)&(neighbors>0.7))])

    #neighbors_index = neighbors_index.reshape(-1)
    neighbors = train.iloc[neighbors_index,:]
    # look for movie in train
    movie = test.iloc[0,1]
    if movie not in df_train_movieIDs.values:
        score = 0.0
        prediction.append(score)
        print("movie not in train: count %d, movie %d" % (counter, movie))
        counter = counter + 1
        continue
    else:
        try:
            movie_index = df_train_movieIDs[train_movieIDs == movie].index[0]
        except:
            print('Except: count %d, movie %d' % (counter, movie))
            print(df_train_movieIDs[train_movieIDs == movie])
        finally:
            counter = counter + 1
        # get scores for the movie
        score_array = neighbors.iloc[:,movie_index]
        non_zero_score_array_index = np.where(score_array != 0)[0]
        #non_zero_score_array_index
        non_zero_score_array = score_array.values[non_zero_score_array_index]
        # get mean score
        if non_zero_score_array.size == 0:
            score = 3.0
        else:
            score = np.mean(non_zero_score_array)

############################################################################
'''

count = test.shape[0]
df_train_movieIDs = pd.DataFrame(train_movieIDs)
df_train_userIDs = pd.DataFrame(train_userIDs)

prediction = []
counter = 0

start_time = time.time()
for i in range(count):
    # get neighbors of the user
    # userID
    userID = test.iloc[i][0]
    # find userID in train
    train_record = train.loc[train.index == userID]
    train_record
    # find neighbors in train_csr
    if userID not in df_train_userIDs.values:
        score = 3.0
        prediction.append(score)
        print("movie not in train: count %d, movie %d" % (counter, movie))
        counter = counter + 1
        continue

    neighbors = corr[userID]
    #neighbors
    neighbors_index = list(neighbors[((neighbors<1.0)&(neighbors>0.95))])

    #neighbors_index = neighbors_index.reshape(-1)
    neighbors = train.iloc[neighbors_index,:]
    # look for movie in train
    movie = test.iloc[i,1]
    if movie not in df_train_movieIDs.values:
        score = 3.0
        prediction.append(score)
        print("movie not in train: count %d, movie %d" % (counter, movie))
        counter = counter + 1
        continue
    else:
        try:
            movie_index = df_train_movieIDs[train_movieIDs == movie].index[0]
        except:
            print('Except: count %d, movie %d' % (counter, movie))
            print(df_train_movieIDs[train_movieIDs == movie])
        finally:
            counter = counter + 1
        # get scores for the movie
        score_array = neighbors.iloc[:,movie_index]
        non_zero_score_array_index = np.where(score_array != 0)[0]
        #non_zero_score_array_index
        non_zero_score_array = score_array.values[non_zero_score_array_index]
        # get mean score
        if non_zero_score_array.size == 0:
            score = 3.0
        else:
            score = np.mean(non_zero_score_array)
    prediction.append(score)
pd.DataFrame(prediction).fillna(0, inplace = True)
print("--- %s seconds ---" % (time.time() - start_time))
prediction

'''
# train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.25, random_state=1)

count = test.shape[0]

# KNN
from sklearn.neighbors import NearestNeighbors
train_model_time = time.time()
neighbor = NearestNeighbors(n_neighbors = 350, metric = 'cosine')
neighbor.fit(train_csr)
print("---Trained model in %s seconds ---" % (time.time() - train_model_time))
'''

'''
# testing algorithm
train
train_csr
#train_csr[3]
neighbor.kneighbors(train_csr[1], return_distance = False)

test
test.iloc[0,0]

neighbors_index = neighbor.kneighbors(train_csr[1], return_distance = False)
neighbors_index = neighbors_index.reshape(-1)
neighbors_index
neighbors = train.iloc[neighbors_index,:]
neighbors
test
movie = test.iloc[0,1]
movie

df_movieIDs = pd.DataFrame(movieIDs)
#movieIDs == movie
movie_index = df_movieIDs[movieIDs == movie].index[0]

score_array = neighbors.iloc[:,movie_index]
score_array
#non_zero_score_array_index = np.nonzero(score_array)
non_zero_score_array_index = np.where(score_array != 0)[0]
non_zero_score_array_index
non_zero_score_array = score_array.values[non_zero_score_array_index]
score = np.mean(non_zero_score_array)
score

train
#train.loc[:,('rating', slice(None))]

#index = train[train['userID']==test.iloc[0,0]]
'''

'''
df_train_movieIDs = pd.DataFrame(train_movieIDs)
#df_train_movieIDs.values#[0]
#int(65037) in df_train_movieIDs.values
#65133 in df_train_movieIDs.values

# predict among similar users
prediction = []
counter = 0

start_time = time.time()
for i in range(count):
    # get neighbors of the user
    # userID
    userID = test.iloc[i][0]
    # find userID in train
    train_record = train.loc[train.index == userID]
    # find neighbors in train_csr
    neighbors_index = neighbor.kneighbors(train_record, return_distance = False)
    #neighbors_index = neighbor.kneighbors(train_csr[i], return_distance = False)
    neighbors_index = neighbors_index.reshape(-1)
    neighbors = train.iloc[neighbors_index,:]
    # look for movie in train
    movie = test.iloc[i,1]
    if movie not in df_train_movieIDs.values:
        score = 0.0
        prediction.append(score)
        print("movie not in train: count %d, movie %d" % (counter, movie))
        counter = counter + 1
        continue
    else:
        try:
            movie_index = df_train_movieIDs[train_movieIDs == movie].index[0]
        except:
            print('Except: count %d, movie %d' % (counter, movie))
            print(df_train_movieIDs[train_movieIDs == movie])
        finally:
            counter = counter + 1
        # get scores for the movie
        score_array = neighbors.iloc[:,movie_index]
        non_zero_score_array_index = np.where(score_array != 0)[0]
        #non_zero_score_array_index
        non_zero_score_array = score_array.values[non_zero_score_array_index]
        # get mean score
        if non_zero_score_array.size == 0:
            score = 3.0
        else:
            score = np.mean(non_zero_score_array)
    prediction.append(score)
pd.DataFrame(prediction).fillna(0, inplace = True)
print("--- %s seconds ---" % (time.time() - start_time))
prediction
'''

'''
# output
#sub = pd.DataFrame(clusters)
prediction = clf.predict(test)
# decoding Y
prediction = lab_enc.inverse_transform(prediction)
prediction.shape
'''
sub = pd.DataFrame(prediction)

#sub
sub = sub[0:]
submission = open("submission.txt","w")
submission.write(sub.to_string(index = False, header = False))
submission.close()
