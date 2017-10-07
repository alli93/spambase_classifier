import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from  sklearn.model_selection  import  train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import KFold

file = open("spambase.data")
data = np.loadtxt(file,delimiter=",")

X = data[:, 0:57]
y = data[:, 57]

dataframe = pd.DataFrame(data=X)

# Normalize and then scale data, necessary because MultinomialNB cannot handle negative values
# Other methods include scaling every attribute to it's min value, or not normalizing the data at all
dataframe_norm = dataframe.apply(lambda x: (x - np.mean(x)) / np.std(x))
dataframe_norm_upscaled = dataframe_norm.apply(lambda x: (x + abs(min(x))))

X_train ,  X_test ,  y_train ,  y_test = train_test_split(dataframe_norm_scaled,  y,  test_size=0.20 ,  random_state=42)

# kFold cross-validation for alpha
splits = 10
kf = KFold(n_splits = splits)

alpha_tests = np.arange(0, 10.0, 0.01)
kFold_train = np.zeros(len(alpha_tests))
kFold_test = np.zeros(len(alpha_tests))
for cv_train_index, cv_test_index in kf.split(X_train):
    X_cv_train = [X_train[i] for i in cv_train_index]
    y_cv_train = [y_train[i] for i in cv_train_index]
    X_cv_test = [X_train[i] for i in cv_test_index]
    y_cv_test = [y_train[i] for i in cv_test_index]
    i = 0
    for test in alpha_tests:
        mNB_cv = MultinomialNB(alpha=test)
        mNB_cv.fit(X_cv_train, y_cv_train)
        kFold_train[i] += mNB_cv.score(X_cv_train, y_cv_train) / splits
        kFold_test[i]  += mNB_cv.score(X_cv_test, y_cv_test) / splits
        i += 1
    
plt.ylabel("Accuracy", fontsize=16)
plt.xlabel("Number of components", fontsize=16)

plt.plot(alpha_tests, kFold_train,  label='Train', color = 'blue')
plt.plot(alpha_tests, kFold_test, label='Test', color = 'red')

plt.legend(bbox_to_anchor=(0.8, 0.3), loc=2, borderaxespad=0.)

def percent(y, x):
   return str(100 * y) + '%'

formatter = FuncFormatter(percent)
plt.gca().yaxis.set_major_formatter(formatter)

plt.show()

