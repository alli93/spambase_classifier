import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection  import  train_test_split

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.ticker import FuncFormatter


# number of components 
n = 48

file = open("spambase.data")
data = np.loadtxt(file,delimiter=",")

X = data[:, 0:57]
y = data[:, 57]

dataframe = pd.DataFrame(data=X)

#apply normalization function to every attribute
dataframe_norm = dataframe.apply(lambda x: (x - np.mean(x)) / np.std(x))

X_train ,  X_test ,  y_train ,  y_test = train_test_split(dataframe_norm,  y,  test_size=0.20 ,  random_state=42)

kNN = KNeighborsClassifier()
kNN.fit(X_train, y_train)

score_train = kNN.score(X_train, y_train)
score_test = kNN.score(X_test, y_test)
print("Pre-reduction Training score: %f, Testing score: %f" %(score_train, score_test))

pca = PCA(n_components = n)
pca.fit(dataframe_norm)
X = pca.transform(dataframe_norm)

X_train ,  X_test ,  y_train ,  y_test = train_test_split(X,  y,  test_size=0.20 ,  random_state=42)

kNN = KNeighborsClassifier()
kNN.fit(X_train, y_train)

score_train = kNN.score(X_train, y_train)
score_test = kNN.score(X_test, y_test)
print("Post-reduction Training score: %f, Testing score: %f" %(score_train, score_test))
