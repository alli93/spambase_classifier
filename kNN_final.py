import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection  import  train_test_split

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import KFold

def pca(n_components, dataframe):
    pca = PCA(n_components = n_components)
    pca.fit(dataframe)
    return pca.transform(dataframe)

# Hypertune parameters
n_components = 20 
k = 13

file = open("spambase.data")
data = np.loadtxt(file,delimiter=",")

X = data[:, 0:57]
y = data[:, 57]

dataframe = pd.DataFrame(data=X)

#apply normalization function to every attribute
dataframe_norm = dataframe.apply(lambda x: (x - np.mean(x)) / np.std(x))

X_train ,  X_test ,  y_train ,  y_test = train_test_split(dataframe_norm,  y,  test_size=0.20 ,  random_state=42)

# kNN without any hypertuning or dimensionality reduction
kNN = KNeighborsClassifier()
kNN.fit(X_train, y_train)

pre_score_train = kNN.score(X_train, y_train)
pre_score_test = kNN.score(X_test, y_test)

# kNN with dimensionality reduction and hypertuning
X = pca(n_components, dataframe_norm)
X_train ,  X_test ,  y_train ,  y_test = train_test_split(X,  y,  test_size=0.20 ,  random_state=42)

kNN = KNeighborsClassifier(n_neighbors = k)
kNN.fit(X_train, y_train)

post_score_train = kNN.score(X_train, y_train)
post_score_test = kNN.score(X_test, y_test)

ind = np.arange(2)
width = 0.2

def percent(y, x):
   return str(100 * y) + '%'

formatter = FuncFormatter(percent)
plt.gca().yaxis.set_major_formatter(formatter)

p1 = plt.bar(ind, [pre_score_train, pre_score_test], width)
p2 = plt.bar(ind+(width*1.1), [post_score_train, post_score_test], width)
plt. xticks(ind+(width/2), ('Train', 'Test'))
plt.legend((p1[0], p2[0]), ('Preliminary', 'Post-tuning'))
plt.ylim(0.85, 1)
plt.show()