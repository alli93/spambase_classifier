"""
To be added to kNN for plotting confusion matrix
"""

y_pred = kNN.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", square=True, xticklabels=['ham','spam'], yticklabels=['ham','spam'], cbar=False)
plt.title('Confusion matrix')
