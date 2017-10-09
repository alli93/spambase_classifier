import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.neural_network import MLPClassifier
from  sklearn.model_selection  import  train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score

file = open("spambase.data")
data = np.loadtxt(file,delimiter=",")

X = data[:, 0:57]
y = data[:, 57]

dataframe = pd.DataFrame(data=X)

dataframe_norm = dataframe.apply(lambda x: (x - np.mean(x)) / np.std(x))

X_train ,  X_test ,  y_train ,  y_test = train_test_split(dataframe_norm,  y,  test_size=0.20 ,  random_state=42)

aNN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

aNN.fit(X_train, y_train)

precision_score(y_test, aNN.predict(X_test), average = 'micro')
