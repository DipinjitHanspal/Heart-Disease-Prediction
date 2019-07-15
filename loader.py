# -*- coding: utf-8 -*-
## API for loading and managing heart disease data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

test_dataset = pd.read_csv('test_values.csv', header=0)
train_dataset_features = pd.read_csv('train_values.csv', header=0).set_index("patient_id")
train_dataset_labels = pd.read_csv('train_labels.csv').set_index("patient_id")
labelencoder = LabelEncoder()
train_dataset_features['thal'] = labelencoder.fit_transform(train_dataset_features['thal'])
features = train_dataset_features.values
labels = train_dataset_labels.values
print(features[1, :], features[3, :])
print(features[1, :], features[3, :])
#onehotencoder = OneHotEncoder()
#features = onehotencoder.fit_transform(features).toarray()
#print(features[1, :], features[3, :])
X_train, X_test, Y_train, Y_test = train_test_split(features, labels)

#
#numerical_float_features = train_dataset_features.dtypes == 'float'
#numerical_int_features = train_dataset_features.dtypes == 'int64'
#categorical_features = ~np.logical_or(numerical_float_features, numerical_int_features) 
#
#preprocess = make_column_transformer(
#        (make_pipeline(SimpleImputer(), StandardScaler()), np.logical_or(numerical_float_features, numerical_int_features)),
#        (make_pipeline(LabelEncoder(), OneHotEncoder()), categorical_features)
#                                     )

f = train_dataset_features.dtypes == 'float'

preprocess = make_column_transformer((make_pipeline(StandardScaler()), np.logical_or(f, ~f)))
#model = make_pipeline(preprocess, MLPClassifier(max_iter=1000, solver='lbfgs'))
#model = make_pipeline(preprocess, SVC())
model = make_pipeline(preprocess, SVC(kernel='rbf', C=1.4))
#model = make_pipeline(preprocess, GaussianNB())
#model = make_pipeline(preprocess, KNeighborsClassifier(n_neighbors='3'))
model.fit(X_train, Y_train.ravel())

print(model.score(X_test, Y_test))