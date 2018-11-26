## Simple classifier implemented using sklearn (cross-validation, grid search,etc.) 

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


# read the training set
dataFrame_tr = DataFrame.from_csv("train.csv")
X = dataFrame_tr[dataFrame_tr.columns[1:17]]
y = dataFrame_tr[dataFrame_tr.columns[0]]

# read the test set
dataFrame_te = DataFrame.from_csv("test.csv")
X_test = dataFrame_te[dataFrame_te.columns[0:16]]

# Normalize/Scale data
X = StandardScaler().fit_transform(X)
X_test = StandardScaler().fit_transform(X_test)

# Split the dataset in ten equal parts
X_train, X_test_split, y_train, y_test = train_test_split(X, y, test_size=0.1)

params = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [
                (10,),(10,6),(10,6,4,),(15,8,7),(10,7,3),(12,6,4),(100,50)
             ],
            'alpha': [1, 0.1, 0.01, 0.001]
        }
       ]

# Use cross-validation to find best tuning parameters
clf = GridSearchCV(MLPClassifier(), params, cv=5, scoring='accuracy',verbose=1)
clf.fit(X_train, y_train)
print("Best parameters set found on development set:")
print(clf.best_params_)

# clf = MLPClassifier(activation='tanh',solver='adam', alpha=0.1, hidden_layer_sizes=(100,50))
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test_split)

# Predict labels
y_pred = clf.predict(X_test_split)


# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print(' Accuracy:%.5f'% (acc))
print(classification_report(y_test, y_pred))

# Predict labels with best_parameters
y_pred_final = clf.predict(X_test)


# Dump results into a *.csv file
data = pd.DataFrame(y_pred_final,index = np.arange(2000, 5000, dtype=np.int), columns = ['y'])
data.index.name = 'Id'
data.to_csv('CVResult.csv')



