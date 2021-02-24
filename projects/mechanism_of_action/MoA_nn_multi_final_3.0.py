# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings
import os
'''
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

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
from sklearn.preprocessing import MinMaxScaler
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
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics import silhouette_score
#from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import f1_score
from math import sqrt
import time
import string
#from nltk.stem import PorterStemmer
#from nltk.corpus import stopwords

warnings.filterwarnings('ignore')

# load data
'''
train_drug = pd.read_csv(r"../input/lish-moa/train_drug.csv", header = 0)
train_features = pd.read_csv(r"../input/lish-moa/train_features.csv", header = 0)
train_targets_nonscored = pd.read_csv(r"../input/lish-moa/train_targets_nonscored.csv", header = 0)
train_targets_scored = pd.read_csv(r"../input/lish-moa/train_targets_scored.csv", header = 0)
test_features = pd.read_csv(r"../input/lish-moa/test_features.csv", header = 0)
sample_submission = pd.read_csv(r"../input/lish-moa/sample_submission.csv", header = 0)
'''
train_drug = pd.read_csv(r"C:/Users/wmwms/OneDrive - George Mason University/Academics/2020Fall/CS584/project/data/train_drug.csv",\
 header = 0)
train_features = pd.read_csv(r"C:/Users/wmwms/OneDrive - George Mason University/Academics/2020Fall/CS584/project/data/train_features.csv",\
 header = 0)#, nrows = 2000)
train_targets_nonscored = pd.read_csv(r"C:/Users/wmwms/OneDrive - George Mason University/Academics/2020Fall/CS584/project/data/train_targets_nonscored.csv",\
 header = 0)
train_targets_scored = pd.read_csv(r"C:/Users/wmwms/OneDrive - George Mason University/Academics/2020Fall/CS584/project/data/train_targets_scored.csv",\
 header = 0)#, nrows = 2000)
test_features = pd.read_csv(r"C:/Users/wmwms/OneDrive - George Mason University/Academics/2020Fall/CS584/project/data/test_features.csv",\
 header = 0)
sample_submission = pd.read_csv(r"C:/Users/wmwms/OneDrive - George Mason University/Academics/2020Fall/CS584/project/data/sample_submission.csv",\
 header = 0)

# data
#train_drug
#train_features
#test_features
#train_targets_nonscored
#train_targets_scored
#sample_submission

train_features['cp_type'].dtypes
columns = train_features.columns
#columns
#[col for col in columns if col.startswith('cp_type')]

# get genes and cells
GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]
CATEGORY = [col for col in train_features.columns if type(col) == 'object']
#CATEGORY
# data prep
y_train = train_targets_scored.copy()
y_train = y_train.drop(labels = ['sig_id'], axis = 1)
x_train = train_features.copy()

y_pred = sample_submission.copy().drop(labels = ['sig_id'], axis = 1)
x_pred = test_features.copy()

train = pd.concat([train_features.copy(), train_targets_scored.copy()], axis = 1)
pred = pd.concat([test_features.copy(), sample_submission.copy()], axis = 1)
train = train.drop(labels = ['sig_id'], axis = 1)
pred = pred.drop(labels = ['sig_id'], axis = 1)
train.shape
pred.shape

train_categories = list(y_train.columns.values)
len(train_categories)
pred_categories = list(y_pred.columns.values)
len(pred_categories)

#pred_categories[pred_categories in train_categories]
#train_categories
#pred_categories

x_train = x_train.drop(labels = ['sig_id'], axis = 1)
x_pred = x_pred.drop(labels = ['sig_id'], axis = 1)
#x_train
#y_train

#train.dtypes
#pred.dtypes
train

train['cp_type'] = LabelEncoder().fit_transform(train['cp_type'])
train['cp_dose'] = LabelEncoder().fit_transform(train['cp_dose'])
train
pred['cp_type'] = LabelEncoder().fit_transform(pred['cp_type'])
pred['cp_dose'] = LabelEncoder().fit_transform(pred['cp_dose'])
pred

x_train['cp_type'] = LabelEncoder().fit_transform(x_train['cp_type'])
x_train['cp_dose'] = LabelEncoder().fit_transform(x_train['cp_dose'])
x_train
x_pred['cp_type'] = LabelEncoder().fit_transform(x_pred['cp_type'])
x_pred['cp_dose'] = LabelEncoder().fit_transform(x_pred['cp_dose'])
#x_train = x_train.apply(LabelEncoder().fit_transform)
#x_pred = x_pred.apply(LabelEncoder().fit_transform)



x_train
#x_train.dtypes
#x_pred.dtypes

# corr
cell_corr = x_train[CELLS].corr()
cell_corr
gene_corr = x_train[GENES].corr()
gene_corr

x_corr = x_train.corr()
x_corr
#list(x_corr[((x_corr < 1.0)&(x_corr>0.9)&(x_corr != 'NaN'))])
y_corr = y_train.corr()
y_corr

# define min max scaler
scaler = MinMaxScaler()
# transform data
x_train = scaler.fit_transform(x_train)
x_pred = scaler.fit_transform(x_pred)



X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.20, random_state=21)

#X_train
#Y_train

#X_test
#Y_test
# kmeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 25, random_state = 0).fit(X_train)
labels = kmeans.labels_
#labels
#kmeans.cluster_centers_
kmeans.cluster_centers_

# PCA
X = X_train.copy()
y = labels
target_names =labels
pca = PCA(n_components=200)
X_r = pca.fit(X).transform(X)
df = pd.DataFrame(X)
df['pca-one'] = X_r[:,0]
df['pca-two'] = X_r[:,1]
df['y'] = y

print('PCA explained variance ratio: %s' % str(pca.explained_variance_ratio_.sum()))

#X, y = None, None
# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 25),
    data=df.iloc[rndperm,:],
    legend="full",
    alpha=0.3
)

X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
x_pred = pca.fit_transform(x_pred)

#from sklearn.preprocessing import MultiLabelBinarizer
#clf = MultiLabelBinarizer().fit_transform(y_train)

from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns

#MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(x_train,y_train).predict(y_test)
predictions = pd.DataFrame()
count = 0
val_scores = []

'''
# KNN
#clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30)
clf = DecisionTreeClassifier()

# RandomForest
forest = RandomForestClassifier(random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
pred = multi_target_forest.fit(X_train, Y_train).predict(X_test)
pred.shape
pred = pd.DataFrame(pred)
pred.describe()
'''
len(pred_categories)

# log loss function
def mean_columnwise_logloss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, (1 - 1e-15))
    score = - np.mean(np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=0))
    return score

scorer = make_scorer(mean_columnwise_logloss, greater_is_better=False)

#sorted(sklearn.metrics.SCORERS.keys())
#x_train
#X_train

'''
# run model
start_time = time.time()
for category in pred_categories:
    print('**Processing {}th feature {} ...**'.format(count, category))
    train_time = time.time()


    #neighbors
    #nbs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(X_train)
    #distances, indices = nbs.kneighbors(X_record)
    #X_new = X_train[indices]
    #Y_new = Y_train[indices]

    #clf = SVM()
    #clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()

    # Training model on train data
    #clf.fit(X_new, Y_new[category])
    clf.fit(X_train, Y_train[category])
    score = cross_validate(clf, X_train, Y_train[category], scoring=scorer)
    print('Cross val score is {}'.format(score))
    #print('Loss: %.3f (%.3f)' % (np.mean(score), np.std(score)))



    # calculating test accuracy
    #train_prediction = clf.predict(X_test)
    #loss = mean_columnwise_logloss(Y_test[category], train_prediction)
    #print('Loss is {}'.format(loss))
    #print('Test score is {}'.format(f1_score(Y_test[category], train_prediction)))


    # prepare the cross-validation procedure
    #cv = KFold(n_splits=5, random_state=1, shuffle=True)
    # evaluate model
    #score = cross_val_score(clf, x_train, y_train, scoring=scorer, cv=cv, n_jobs=-1)
    # report performance
    #print('Cross validation Log Loss: %.3f (%.3f)' % (mean(score), std(score)))
    val_scores.append(score)

    #prediction = clf.predict(x_pred)
    #predictions[category] = prediction

    count = count + 1
    print('time usage {}'.format(time.time()-train_time))
    print("\n")
print("--- %s seconds ---" % (time.time() - start_time))
'''

# neural network
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf = MultiOutputClassifier(nn)

# MultiOutputClassifier
#clf = MultiOutputClassifier(XGBClassifier())#tree_method='gpu_hist'))
#clf.get_params().keys()
'''
params = {'estimator__colsample_bytree': 0.6522,
          'estimator__gamma': 3.6975,
          'estimator__learning_rate': 0.0503,
          'estimator__max_delta_step': 2.0706,
          'estimator__max_depth': 10,
          'estimator__min_child_weight': 31.5800,
          'estimator__n_estimators': 166,
          'estimator__subsample': 0.8639
         }

_ = clf.set_params(**params)
'''

start_time = time.time()

#print('**Processing {}th feature {} ...**'.format(count, category))
train_time = time.time()

clf.fit(X_train, Y_train)
score = cross_validate(clf, X_train, Y_train, scoring=scorer)
print('Cross val score is {}'.format(score))
    #print('Loss: %.3f (%.3f)' % (np.mean(score), np.std(score)))



    # calculating test accuracy
    #train_prediction = clf.predict(X_test)
    #loss = mean_columnwise_logloss(Y_test[category], train_prediction)
    #print('Loss is {}'.format(loss))
    #print('Test score is {}'.format(f1_score(Y_test[category], train_prediction)))


    # prepare the cross-validation procedure
    #cv = KFold(n_splits=5, random_state=1, shuffle=True)
    # evaluate model
    #score = cross_val_score(clf, x_train, y_train, scoring=scorer, cv=cv, n_jobs=-1)
    # report performance
    #print('Cross validation Log Loss: %.3f (%.3f)' % (mean(score), std(score)))
val_scores.append(score)

    #prediction = clf.predict(x_pred)
    #predictions[category] = prediction

    #count = count + 1
print('time usage {}'.format(time.time()-train_time))
print("\n")
print("--- %s seconds ---" % (time.time() - start_time))


#predictions.values
val_scores = pd.DataFrame(val_scores)
#val_scores
np.mean(val_scores['test_score'])

plt.figure(figsize = (15, 10))
plt.scatter(range(5), np.mean(val_scores['test_score']))
plt.plot(range(5), np.mean(val_scores['test_score']))
plt.show()

# prediction
#sub = pd.DataFrame(predictions)
#sub.to_csv('submission.csv')#, index=False)


#######################################################
