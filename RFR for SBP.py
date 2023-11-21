import numpy as np
from array import array
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import time
import matplotlib.dates as md
import pylab as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import pydot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

p=pd.read_csv(r'C://BP vital sign//vitalsign_data//Eyes Nose p.csv')
p=pd.DataFrame(p)
p= p.iloc[: , 1:]
#p.dropna()
p=p.apply(lambda row: row.fillna(row.mean()), axis=1)
#p= pd.DataFrame(np.transpose(p))
#print(p)

################################### RFR for SBP #######################################
# Labels are the values we want to predict
labels = np.array(p['SBP'])
# Remove the labels from the features
# axis 1 refers to the columns
features= p.drop('SBP', axis = 1)
features= p.drop('DBP', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

"""
# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('SBP')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))
"""
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
predictions_train = rf.predict(train_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'for SBP')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
 # Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')

from IPython.display import Image
Image(filename = 'tree.png')
print(predictions)
print(test_labels)
print(train_labels)
print(predictions_train)

#means=mean_squared_error(predictions, test_labels)
#print('Mean Squared Error:', round(np.mean(means), 2), 'for SBP')
abb=[predictions,test_labels]
print(abb)
print(np.std(abb))

