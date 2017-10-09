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

def pca(n_components, dataframe):
    pca = PCA(n_components = n_components)
    pca.fit(dataframe)
    return pca.transform(dataframe)

n_components = 14 

file = open("spambase.data")
data = np.loadtxt(file,delimiter=",")

X = data[:, 0:57]
y = data[:, 57]

dataframe = pd.DataFrame(data=X)

#apply normalization function to every attribute
dataframe_norm = dataframe.apply(lambda x: (x - np.mean(x)) / np.std(x))

# Hypertune k
X = pca(n_components, dataframe_norm)
X_train ,  X_test ,  y_train ,  y_test = train_test_split(X,  y,  test_size=0.20 ,  random_state=42)

# First guess. Lross-validate on these values and plot out a curve
k_tests = list(range(1, 100))
k_precision_train = np.zeros(len(k_tests))
k_precision_test = np.zeros(len(k_tests))

splits = 10
kf = KFold(n_splits = splits)
for cv_train_index, cv_test_index in kf.split(X_train):
        X_cv_train = [X_train[i] for i in cv_train_index]
        y_cv_train = [y_train[i] for i in cv_train_index]
        X_cv_test = [X_train[i] for i in cv_test_index]
        y_cv_test = [y_train[i] for i in cv_test_index]
        i=0
        for test in k_tests:
            kNN = KNeighborsClassifier(n_neighbors = test)
            kNN.fit(X_cv_train, y_cv_train) 
            k_precision_train [i-1] += precision_score(y_cv_train, kNN.predict(X_cv_train), average = 'micro') / splits
            k_precision_test [i-1]  += precision_score(y_cv_test, kNN.predict(X_cv_test), average = 'micro') / splits
            i += 1

plt.ylabel("Precision", fontsize=16)
plt.xlabel("K-Nearest", fontsize=16)

plt.plot(k_tests[0:95], k_precision_train[0:95] , label='Train', color = 'blue')
plt.plot(k_tests[0:95], k_precision_test[0:95] , label='Test', color = 'red')

plt.legend(bbox_to_anchor=(0.8, 0.5), loc=2, borderaxespad=0.)

def percent(y, x):
   return str(100 * y) + '%'

formatter = FuncFormatter(percent)
plt.gca().yaxis.set_major_formatter(formatter)

plt.show()
