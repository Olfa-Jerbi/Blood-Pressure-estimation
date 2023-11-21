from turtle import shape
import numpy as np
from array import array
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import matplotlib.dates as md
from datetime import datetime
from pathlib import Path
import padasip as pa
import fir1
from fir1 import Fir1
from scipy import signal
from scipy.signal import kaiserord, lfilter, firwin, freqz
from sklearn.preprocessing import scale 
from scipy import interpolate
from scipy.signal import argrelextrema
from scipy.fft import fft, fftfreq
from scipy.fft import rfft, rfftfreq
import numpy as np
from array import array
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
import joblib
########## After the preprocessing as done by Hyunwoo, save as csv, modify file path

#pd.DataFrame(detrending_data_bp).to_csv("C://BP vital sign//vitalsign_data//TPR12_yj//re_forehead_final pre v2.csv", index=None,line_terminator='\n')
data =pd.read_csv(r'C://BP vital sign//vitalsign_data//TPR2_hw//re_Nose_final pre.csv')
data=pd.DataFrame(data) 
data_cut=np.transpose(data)
########## Choose portion from signal, in seconds * 30 (fps)
data_cut=data_cut.iloc[:, 1700:2000]
SAMPLE_RATE=30
DURATION=10
def FFT_features(data_cut,SAMPLE_RATE,DURATION):
    N = SAMPLE_RATE * DURATION
    df = pd.DataFrame(index=range(((SAMPLE_RATE * DURATION)//2)+1),columns=range(14))
    for i in range(0,14):
     data1=data_cut.iloc[i,:]
     data1=data1.to_numpy()
     yf = rfft(data1)
     yf=np.abs(yf)
     xf = rfftfreq(N, 1 / SAMPLE_RATE)
     df[i]=yf
    #print(df)
    return df

##### Call function
FFT_features(data_cut,30,10)


#################### Train and save Random forest algorithms#################################
###########  For SBP
############ Choose which dataset to train with
p=pd.read_csv(r'C://BP vital sign//vitalsign_data//Eyes Nose p.csv')
p=pd.DataFrame(p)
p= p.iloc[: , 1:]
#p.dropna()
p=p.apply(lambda row: row.fillna(row.mean()), axis=1)
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

#print('Training Features Shape:', train_features.shape)
#print('Training Labels Shape:', train_labels.shape)
#print('Testing Features Shape:', test_features.shape)
#print('Testing Labels Shape:', test_labels.shape)


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

joblib.dump(rf,'C:/BP vital sign/rfr_SBP.joblib')
loaded_rfr_SBP=joblib.load('C:/BP vital sign/rfr_SBP.joblib')

########## predict on newly collected data
x=[]
print(loaded_rfr_SBP.predict(x))


###########  For DBP
############ Choose which dataset to train with
p=pd.read_csv(r'C://BP vital sign//vitalsign_data//Eyes Nose p.csv')
p=pd.DataFrame(p)
p= p.iloc[: , 1:]
#p.dropna()
p=p.apply(lambda row: row.fillna(row.mean()), axis=1)
# Labels are the values we want to predict
labels = np.array(p['DBP'])
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

#print('Training Features Shape:', train_features.shape)
#print('Training Labels Shape:', train_labels.shape)
#print('Testing Features Shape:', test_features.shape)
#print('Testing Labels Shape:', test_labels.shape)


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

joblib.dump(rf,'C:/BP vital sign/rfr_DBP.joblib')
loaded_rfr_DBP=joblib.load('C:/BP vital sign/rfr_DBP.joblib')

########## predict on newly collected data 
u=[]
print(loaded_rfr_DBP.predict(u))

########## predict on newly collected data Example
x=pd.read_csv(r'C://BP vital sign//vitalsign_data//test.csv')
x= x.iloc[: , 1:]
#p.dropna()
x=x.apply(lambda row: row.fillna(row.mean()), axis=1)
#print(x)
print(loaded_rfr_DBP.predict(x))