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

n_components = 48
n_attributes = 57


file = open("spambase.data")
data = np.loadtxt(file,delimiter=",")

X = data[:, 0:57]
y = data[:, 57]

dataframe = pd.DataFrame(data=X)

#apply normalization function to every attribute
dataframe_norm = dataframe.apply(lambda x: (x - np.mean(x)) / np.std(x))

dataframe_norm.describe().round(2).loc[['min', 'max','mean','std']]


X_train ,  X_test ,  y_train ,  y_test = train_test_split(dataframe_norm,  y,  test_size=0.20 ,  random_state=42)

kNN = KNeighborsClassifier()
kNN.fit(X_train, y_train)

score_train = kNN.score(X_train, y_train)
score_test = kNN.score(X_test, y_test)
print("Pre-reduction Training score: %f, Testing score: %f" %(score_train, score_test))

X = pca(n_components, dataframe_norm)

X_train ,  X_test ,  y_train ,  y_test = train_test_split(X,  y,  test_size=0.20 ,  random_state=42)

kNN = KNeighborsClassifier()
kNN.fit(X_train, y_train)

score_train = kNN.score(X_train, y_train)
score_test = kNN.score(X_test, y_test)
print("Post-reduction Training score: %f, Testing score: %f" %(score_train, score_test))

# n_components hyperparameter tuning using KFold cross-validation
splits = 10
kf = KFold(n_splits = splits)
kFold_train_score = np.zeros(n_attributes-1)
kFold_test_score = np.zeros(n_attributes-1)
kFold_train_precision = np.zeros(n_attributes-1)
kFold_test_precision = np.zeros(n_attributes-1)
for i in range(1, n_attributes):
    X = pca(i, dataframe)
    # Using the same random_state ensures we do not contaminate our training data with our test data
    X_train ,  X_test ,  y_train ,  y_test = train_test_split(X,  y,  test_size=0.20 ,  random_state=42)
    for cv_train_index, cv_test_index in kf.split(X_train):
        X_cv_train = [X_train[i] for i in cv_train_index]
        y_cv_train = [y_train[i] for i in cv_train_index]
        X_cv_test = [X_train[i] for i in cv_test_index]
        y_cv_test = [y_train[i] for i in cv_test_index]
        kNN = KNeighborsClassifier()
        kNN.fit(X_cv_train, y_cv_train) 
        kFold_train_precision[i-1] += precision_score(y_cv_train, kNN.predict(X_cv_train), average = 'micro') / splits
        kFold_test_precision[i-1]  += precision_score(y_cv_test, kNN.predict(X_cv_test), average = 'micro') / splits
       
plt.ylabel("Precision", fontsize=16)
plt.xlabel("Number of components", fontsize=16)

plt.plot(kFold_train_precision[0:57], label='Train', color = 'blue')
plt.plot(kFold_test_precision[0:57], label='Test', color = 'red')

plt.legend(bbox_to_anchor=(0.8, 0.3), loc=2, borderaxespad=0.)

def percent(y, x):
   return str(100 * y) + '%'

formatter = FuncFormatter(percent)
plt.gca().yaxis.set_major_formatter(formatter)

plt.show()
