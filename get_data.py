import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

file = open("spambase.data")
data = pd.read_table(file, delimiter=",", header=None)

