from turtle import shape
import numpy as np
from array import array
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import time
import matplotlib.dates as md
from datetime import datetime
from pathlib import Path
import padasip as pa
import pylab as pl
import fir1
from fir1 import Fir1
from scipy import signal
#from scipy.interpolate import interp1d
from scipy.signal import kaiserord, lfilter, firwin, freqz
from sklearn.preprocessing import scale 
from scipy import interpolate
#from mpmath import *
from scipy.signal import argrelextrema
#data = np.load('C://BP vital sign//vitalsign_data//TPR11_olfa//re_forehead_human.npy')

data1 =pd.read_csv(r'C://BP vital sign//vitalsign_data//TPR12_yj//re_forehead_final pre.csv')
data1=pd.DataFrame(data1)
#print(data)
data_cut=np.transpose(data1)

####################### CHOOSE TIME HERE number in seconds * 30 * 30 ##################""""""
data_cut=data_cut.iloc[:, 1700:2300]
#print(data_cut)
#plt.plot(np.transpose(data_cut))
#plt.show()
#print(data_cut.iloc[1, :])

xs = [0]
df1=[]
for i in range(0,14):
    
    for r in data_cut.iloc[i, :]:
        
        xs.append(xs[-1] * 0.9 + r)
        df = pd.DataFrame(xs, columns=['data'])

        n = 5  # number of points to be checked before and after

# Find local peaks

        df['min'] = df.iloc[argrelextrema(df.data.values, np.less_equal,
                    order=n)[0]]['data']
                 
        df['max'] = df.iloc[argrelextrema(df.data.values, np.greater_equal,
                   order=n)[0]]['data']
    #df=df.dropna()
    #print(df)
    #print(df)
        df1.append(df)  
        #df2 = pd.concat(df1)
    #print(df1) 
# Plot results
    
#plt.scatter(df.index, df['min'], c='r')
#plt.scatter(df.index, df['max'], c='g')
#plt.plot(df.index, df['data'])
#plt.show()
df2=df1[8399]
#print(shape(df1))
#arr = np.array(df2)
#print(shape(arr))
#print(arr)
#print(type(arr))
#max=arr[:,3]
#print(max)
#df1=pd.DataFrame(df1,columns =['data', 'min', 'max'])
#print(type(df1))
#df1 = df1[pd.notnull(df1['max'])]
#df1 = df1[pd.notnull(df1['min'])]
#df1.columns =['data', 'min', 'max']
#print(df1)


df3=df2[['min']]
df3=df3.dropna()
#print(df3)
df4=df2['max']
df4=df4.dropna()
#print(df4)
#df3=df1.iloc[:,3]
#df3=df3.dropna()
#print(df2)
#print(df3)
pd.DataFrame(df3).to_csv("C://BP vital sign//vitalsign_data//TPR12_yj//re_forehead_min T 20.csv", index=None,line_terminator='\n')

pd.DataFrame(df4).to_csv("C://BP vital sign//vitalsign_data//TPR12_yj//re_forehead_max T 20.csv", index=None,line_terminator='\n')
    
    
    