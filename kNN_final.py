import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection  import  train_test_split

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score

from sklearn.metrics import confusion_matrix


def pca(n_components, dataframe):
    pca = PCA(n_components = n_components)
    pca.fit(dataframe)
    return pca.transform(dataframe)

# Hypertune parameters
n_components = 14
k = 4

file = open("spambase.data")
data = np.loadtxt(file,delimiter=",")

X = data[:, 0:57]
y = data[:, 57]

dataframe = pd.DataFrame(data=X)

#apply normalization function to every attribute
dataframe_norm = dataframe.apply(lambda x: (x - np.mean(x)) / np.std(x))

X_train ,  X_test ,  y_train ,  y_test = train_test_split(dataframe_norm,  y, test_size=0.20 , random_state=42)

# kNN without any hypertuning or dimensionality reduction
kNN = KNeighborsClassifier()
kNN.fit(X_train, y_train)

precision_train_pre = precision_score(y_train, kNN.predict(X_train), average = 'micro')
precision_test_pre = precision_score(y_test, kNN.predict(X_test), average = 'micro')

# kNN with dimensionality reduction and hypertuning
X = pca(n_components, dataframe_norm)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20 , random_state=42)

kNN = KNeighborsClassifier(n_neighbors = k)
kNN.fit(X_train, y_train)

precision_train_post = precision_score(y_train, kNN.predict(X_train), average = 'micro')
precision_test_post = precision_score(y_test, kNN.predict(X_test), average = 'micro')

ind = np.arange(2)
width = 0.2

def percent(y, x):
   return str(100 * y) + '%'

formatter = FuncFormatter(percent)
plt.gca().yaxis.set_major_formatter(formatter)

p1 = plt.bar(ind, [precision_train_pre, precision_test_pre], width)
p2 = plt.bar(ind+(width*1.1), [precision_train_post, precision_test_post], width)
plt.ylabel("Precision", fontsize=16)
plt. xticks(ind+(width/2), ('Train', 'Test'))
plt.legend((p1[0], p2[0]), ('Preliminary', 'Post-tuning'))
plt.ylim(0.85, 1)
plt.show()
