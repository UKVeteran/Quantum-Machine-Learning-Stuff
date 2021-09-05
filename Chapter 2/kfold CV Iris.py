#Import Libraries
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm    # import Support Vector Machines (SVM) for classification
from sklearn import datasets


iris = datasets.load_iris() # load the data

# Data split into train/test sets with 25% reserved for testing
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)

# Linear Support Vector Classifier (SVC) model for prediction using training data
svc = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

# Performance measurement with the test data
svc.score(X_test, y_test)


#cross_val_score a model, the data set, and the number of folds:
cvs = cross_val_score(svc, iris.data, iris.target, cv=5)

# Print accuracy of each fold:
print(cvs)

# Mean accuracy of all 5 folds:
print(cvs.mean())

## try a different polynomial kernel "poly"
svc = svm.SVC(kernel='poly', C=1).fit(X_train, y_train)
cvs = cross_val_score(svc, iris.data, iris.target, cv=5)
print(cvs)
print(cvs.mean())

# Build an SVC model for predicting iris classifications using training data
svc = svm.SVC(kernel='poly', C=1).fit(X_train, y_train)

# Now measure its performance with the test data
svc.score(X_test, y_test)

