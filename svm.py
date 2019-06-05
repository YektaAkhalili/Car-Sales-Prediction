import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data_Forecast.csv")
# print(dataset.head())

# Plot Things
# Red dots are greater than mean value
colors = np.where(dataset["Greater_Than_Mean"] == 1, 'r', 'g')
# dataset.plot.scatter(x='index', y='Sales data', c=colors)
# plt.show()


# attributes and lables
x = dataset.drop('Greater_Than_Mean', axis=1)
y = dataset['Greater_Than_Mean']

# split data into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

# fit model
from sklearn.svm import SVC
clf = SVC(kernel ='linear')
clf.fit(x_train, y_train)

# predictions
y_pred = clf.predict(x_test)
print("Predicted Values: ", y_pred)
# save predictions into file

# evaluating the algorithm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Kernel SVM
print("Different Kernels: ")
# print("Gaussian Kernel: ")
# clf_rbf = SVC(kernel='rbf')
# clf_rbf.fit(x_train, y_train)

# y_pred_g = clf_rbf.predict(x_test)
# print(accuracy_score(y_test, y_pred_g)) #63%

# clf_sig = SVC(kernel = 'sigmoid')
# clf_sig.fit(x_train, y_train)
# y_pred_sig = clf_sig.predict(x_test)
# print(accuracy_score(y_test, y_pred_sig)) #63%
