import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
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

# Different preprocessings
dataframe = pd.DataFrame(data=X)
dataframe_norm = dataframe.apply(lambda x: (x - np.mean(x)) / np.std(x))
min_max_scaler = preprocessing.MinMaxScaler()
dataframe_norm_scaled = min_max_scaler.fit_transform(dataframe_norm)
dataframe_norm_upscaled = dataframe_norm.apply(lambda x: (x + abs(min(x))))

# No preprocessing of the data
X_train ,  X_test ,  y_train ,  y_test = train_test_split(dataframe,  y,  test_size=0.20 ,  random_state=42)
mNB = MultinomialNB()
mNB.fit(X_train, y_train)
score_train = mNB.score(X_train, y_train)
score_test = mNB.score(X_test, y_test)

# Upscaled normalized data with no negative attributes
X_train ,  X_test ,  y_train ,  y_test = train_test_split(dataframe_norm_upscaled,  y,  test_size=0.20 ,  random_state=42)
mNB_upscaled = MultinomialNB()
mNB_upscaled.fit(X_train, y_train)
norm_upscaled_score_train = mNB_upscaled.score(X_train, y_train)
norm_upscaled_score_test = mNB_upscaled.score(X_test, y_test)

# Scaled normalized data 
min_max_scaler = preprocessing.MinMaxScaler()
X_train ,  X_test ,  y_train ,  y_test = train_test_split(dataframe_norm_scaled,  y,  test_size=0.20 ,  random_state=42)
mNB_scaled = MultinomialNB()
mNB_scaled.fit(X_train, y_train)
norm_scaled_score_train = mNB_scaled.score(X_train, y_train)
norm_scaled_score_test = mNB_scaled.score(X_test, y_test)

ind = np.arange(2)
width = 0.2

p1 = plt.bar(ind, [score_train, score_test], width)
p2 = plt.bar(ind+(width*1.1), [norm_upscaled_score_train, norm_upscaled_score_test], width)
p3 = plt.bar(ind+((2*width)*1.1), [norm_scaled_score_train, norm_scaled_score_test], width)

def percent(y, x):
   return str(100 * y) + '%'

formatter = FuncFormatter(percent)
plt.gca().yaxis.set_major_formatter(formatter)

plt. xticks(ind+(width/2), ('Train', 'Test'))
plt.legend((p1[0], p2[0], p3[0]), ('No preprocessing', 'Normalized and upscaled', 'Normalized and scaled'))
plt.ylim(0.6, 1)
plt.show()

