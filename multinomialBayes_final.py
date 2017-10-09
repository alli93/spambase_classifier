import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from  sklearn.model_selection  import  train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.ticker import FuncFormatter

def pca(n_components, dataframe):
    pca = PCA(n_components = n_components)
    pca.fit(dataframe)
    return pca.transform(dataframe)

file = open("spambase.data")
data = np.loadtxt(file,delimiter=",")

X = data[:, 0:57]
y = data[:, 57]

dataframe = pd.DataFrame(data=X)
dataframe_norm = dataframe.apply(lambda x: (x - np.mean(x)) / np.std(x))
dataframe_norm_upscaled = dataframe_norm.apply(lambda x: (x + abs(min(x))))

# Hyperparameters
a = 0.01

# No preprocessing or tuning
X_train ,  X_test ,  y_train ,  y_test = train_test_split(dataframe,  y,  test_size=0.20 ,  random_state=42)
mNB = MultinomialNB()
mNB.fit(X_train, y_train)
untuned_precision_train = precision_score(y_train, mNB.predict(X_train), average = 'micro')
untuned_precision_test = precision_score(y_test, mNB.predict(X_test), average = 'micro')

# Scaled normalized data with hyperparameter tuning
min_max_scaler = preprocessing.MinMaxScaler()
dataframe_norm = dataframe.apply(lambda x: (x - np.mean(x)) / np.std(x))
dataframe_norm_scaled = min_max_scaler.fit_transform(dataframe_norm)
X_train ,  X_test ,  y_train ,  y_test = train_test_split(dataframe_norm_scaled,  y,  test_size=0.20 ,  random_state=42)

mNB_tuned = MultinomialNB(alpha = a)
mNB_tuned.fit(X_train, y_train)
tuned_precision_train = precision_score(y_train, mNB_tuned.predict(X_train), average = 'micro')
tuned_precision_test = precision_score(y_test, mNB_tuned.predict(X_test), average = 'micro')

ind = np.arange(2)
width = 0.2

p1 = plt.bar(ind, [untuned_precision_train, untuned_precision_test], width)
p2 = plt.bar(ind+(width*1.1), [tuned_precision_train, tuned_precision_test], width)

def percent(y, x):
   return str(100 * y) + '%'

formatter = FuncFormatter(percent)
plt.gca().yaxis.set_major_formatter(formatter)
plt.ylabel("Precision", fontsize=16)
plt. xticks(ind+(width/2), ('Train', 'Test'))
plt.legend((p1[0], p2[0]), ('No preprocessing, hypertuning', 'Normalized, scaled, hypertuned'))
plt.ylim(0.6, 1)
plt.show()

