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

splits = 10

X = data[:, 0:57]
y = data[:, 57]

dataframe = pd.DataFrame(data=X)
dataframe_norm = dataframe.apply(lambda x: (x - np.mean(x)) / np.std(x))
X_train ,  X_test ,  y_train ,  y_test = train_test_split(dataframe_norm,  y,  test_size=0.20 ,  random_state=42)

tests = np.arange(0.0005, 1, 0.0005)
learning_rate_precision_train = np.zeros(len(tests))
learning_rate_precision_test = np.zeros(len(tests))

kf = KFold(n_splits = splits)
for cv_train_index, cv_test_index in kf.split(X_train):
        X_cv_train = [X_train[i] for i in cv_train_index]
        y_cv_train = [y_train[i] for i in cv_train_index]
        X_cv_test = [X_train[i] for i in cv_test_index]
        y_cv_test = [y_train[i] for i in cv_test_index]
        i=0
        for test in tests:
            aNN = MLPClassifier(solver='sgd', max_iter = 1000)
            aNN.fit(X_cv_train, y_cv_train)
            learning_rate_precision_train[i-1] += precision_score(y_cv_train, kNN.predict(X_cv_train), average = 'micro') / splits
            learning_rate_precision_test[i-1]  += precision_score(y_cv_test, kNN.predict(X_cv_test), average = 'micro') / splits
            i += 1

plt.ylabel("Precision", fontsize=16)
plt.xlabel("Learning rate", fontsize=16)

plt.plot(tests, learning_rate_precision_train , label='Train', color = 'blue')
plt.plot(tests, learning_rate_precision_test, label='Test', color = 'red')

plt.legend(bbox_to_anchor=(0.8, 0.5), loc=2, borderaxespad=0.)

def percent(y, x):
   return str(100 * y) + '%'

formatter = FuncFormatter(percent)
plt.gca().yaxis.set_major_formatter(formatter)

plt.show()

