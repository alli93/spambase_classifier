import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.ticker import FuncFormatter

file = open("spambase.data")
data = np.loadtxt(file,delimiter=",")

X = data[:, 0:57]
Y = data[:, 0]

dataframe = pd.DataFrame(data=X)

#apply normalization function to every attribute
dataframe_norm = dataframe.apply(lambda x: (x - np.mean(x)) / np.std(x))

#Test different values of n_components
EVC = []
for attribute in dataframe_norm:
    pca = PCA(n_components=attribute)
    pca.fit(dataframe_norm)
    EVC.append(pca.explained_variance_ratio_.sum())

def percent(y, x):
   return str(100 * y) + '%' 

plt.plot(EVC)

formatter = FuncFormatter(percent)
plt.xlabel("Number of Components", fontsize=16)
plt.ylabel("Explained Variance Ratio", fontsize=16)
plt.gca().yaxis.set_major_formatter(formatter)
t=48
plt.annotate('n = 48',
             xy = (t, EVC[t]),
             xycoords='data',
             xytext=(+6, -30),
             textcoords='offset points',
             fontsize=16,)

plt.plot([t, t], [0, EVC[t]], color='blue', linewidth=2.5, linestyle="--")
plt.scatter([t, ], [EVC[t], ], 50, color='blue')
plt.show()

