import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

file = open("spambase.data")
data = np.loadtxt(file, delimiter=",")

X = data[:,0:48]
Y = data[:,57]

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

print(X_r)
