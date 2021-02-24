# this is the python implementation of K-Means Clustering part2
# with silhouette_score
# with PCA
# input[10] 0.55, [5000] 0.54, 0.53
# with kmeans++ without bisecting 0.54
# cosine 0.54
# kmeans++ 0.56
# unit norm 0.52
import os
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
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from scipy.sparse import csr_matrix
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import f1_score
from math import sqrt
import time
import string
#from nltk.stem import PorterStemmer
#from nltk.corpus import stopwords

# set environment
os.chdir(r'C:\Users\wmwms\OneDrive - George Mason University\Academics\2020Fall\CS584\HW3\part2')
os.getcwd()

# load data
data = pd.read_csv(r"test.txt", sep = ",", header = None)#, header = 0)#, nrows = 2000)
#data.head
data[0].shape
labels = data.columns

plt.spy(data_array)
plt.title("sparse matrix")

# normalize
data_array = data.values
data_array = preprocessing.normalize(data_array, norm = 'l2')


#nor = Normalizer()
#nor.fit(data_array)
#data_array = nor.transform(data_array)
#sc = StandardScaler()
#sc.fit(data_array)
#data_array = sc.transform(data_array)

# sparse matrix
#data_array = csr_matrix(data_array)

#####################################################################################

def kmeans(input, k, maxrounds):
    # randomly choose k numbers as starting index
    #index = np.random.choice(len(input), k, replace=False)
    # choose initial centroids based on the index
    #centroids = input[index, :]

    # use kmeans++ to initiate centroids
    centroids = initiate_centroids_plus(input, k)

    # calculate min euclidean distance between each point and each centroid
    # find the nearest centroid based on the min distance, return its cluster
    cluster = np.argmin(distance.cdist(input, centroids, 'cityblock'),axis=1)
    # within max rounds calculate new centroids
    error = 0
    for i in range(maxrounds):
        # calculate new centroids
        centroids = np.vstack([input[cluster==j,:].mean(axis=0) for j in range(k)])
        # calculate new clusters
        temp = np.argmin(distance.cdist(input, centroids, 'cityblock'),axis=1)
        # calculate error
        #error += euclidean_distances(temp - cluster)
        error += np.linalg.norm(temp - cluster)
        print("%d round, error: %d" % (i,error))
        # if the error does not increase and centroids do not change, return
        if np.array_equal(cluster,temp):break
        # otherwise assign new clusters
        cluster = temp
    #print('created %d clusters' % len(cluster) )
    return cluster

# use kmeans++ to initiate centroids
def initiate_centroids_plus(input, k):
    # arbitrarily pick first centroid
    centroids = [input[1000]]
    # loop until k centroids are picked
    for i in range(1, k):
        # calculate distance array betweem each point and each centroid
        distance = np.array([min([np.inner(c-x, c-x) for c in centroids]) for x in input])
        # calculate weighted probability distribution
        probabilty = distance/distance.sum()
        cumulative_probability = probabilty.cumsum()
        rand = np.random.rand()
        # choose one new point as centroid based on the weighted probability
        for j, p in enumerate(cumulative_probability):
            if rand < p:
                i = j
                break
        centroids.append(input[i])
        #print("centroids: ", centroids)
    return np.array(centroids)

# helper function to calculate sse within cluster
def sse(cluster):
    centroid = np.mean(cluster,0)
    errors = np.linalg.norm(cluster - centroid, ord = 2, axis = 1)
    return np.sum(errors)

# bisecting kmeans algorithm
def bisecting_kmeans(input, b = 2, maxrounds = 100):
    # start with one whole cluster
    clusters = [input]
    # each iteration breaks one cluster into two until reach k clusters
    n = 1
    while n <= k:
        print("bisecting round %d, clusters %d" % (n,len(clusters)))
        # choose the cluster with largest sse to break
        max_sse_cluster = np.argmax([sse(c) for c in clusters])
        cluster = clusters.pop(max_sse_cluster)
        print("max sse: %d" % sse(cluster))

        # use Silhouette_score to break
        #min_sil_cluster = np.argmin([silhouette_score(c,labels) for c in clusters])
        #cluster = clusters.pop(min_sil_cluster)

        # break the cluster into 2
        new_clusters = kmeans(cluster, 2, maxrounds)
        # append the new clusters to the original cluster
        clusters.extend(new_clusters)
        n = n+1
    return clusters

#####################################################################################

# parameters
k = 10
maxrounds = 150

# test data
start_time = time.time()
#clusters = bisecting_kmeans(data_array, 10, 100)
clusters = kmeans(data_array, k, maxrounds)

'''
#internal metric
for i in range(2, 21, 2):
    kmeans(data_array, i, maxrounds)
    print("k: %d" % i)

plt.figure(figsize = (15, 10))
plt.plot(range(2, 21, 2), [109, 1261, 1059, 1482, 2036, 1851, 4626, 3214, 2454, 3759])
plt.show()
'''

print("--- %s seconds ---" % (time.time() - start_time))
# assert clusters size unchanged
assert len(data_array) == len(clusters)
#len(clusters)

#####################################################################################

# PCA
X = data_array.copy()
y = clusters
target_names =labels
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
df = pd.DataFrame(X)
df['pca-one'] = X_r[:,0]
df['pca-two'] = X_r[:,1]
df['y'] = y

print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
X, y = None, None
# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[rndperm,:]["pca-one"],
    ys=df.loc[rndperm,:]["pca-two"],
    c=df.loc[rndperm,:]["y"],
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
plt.show()

'''
plt.figure()
colors = ['navy', 'turquoise', 'darkorange','red','blue','green','yellow','orange','gray','purple']
lw = 2

for color, i, target_name in zip(colors, range(0,10), target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of image dataset')
plt.show()
'''

'''
# t-SNE
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_array)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

data_array_tsne = pd.DataFrame(data_array.copy())
data_array_tsne['tsne-2d-one'] = tsne_results[:,0]
data_array_tsne['tsne-2d-two'] = tsne_results[:,1]
y = clusters
data_array_tsne['y'] = y
y = None

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=data_array_tsne,
    legend="full",
    alpha=0.3
)
'''

# PCA-> tSNE
data_array_pca_tsne = pd.DataFrame(data_array.copy())

pca_100 = PCA(n_components=100)
pca_result_100 = pca_100.fit_transform(data_array_pca_tsne)
print('Cumulative explained variation for 100 principal components: {}'.format(np.sum(pca_100.explained_variance_ratio_)))

time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_100)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

y = kmeans(tsne_pca_results,10,150)
data_array_pca_tsne['tsne-pca100-one'] = tsne_pca_results[:,0]
data_array_pca_tsne['tsne-pca100-two'] = tsne_pca_results[:,1]

data_array_pca_tsne['y'] = y

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-pca100-one", y="tsne-pca100-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=data_array_pca_tsne,
    legend="full",
    alpha=0.3
)

#####################################################################################

# output
#sub = pd.DataFrame(clusters)
sub = pd.DataFrame(y)

#sub
sub = sub[0:]
submission = open("submission.txt","w")
submission.write(sub.to_string(index = False, header = False))
submission.close()
