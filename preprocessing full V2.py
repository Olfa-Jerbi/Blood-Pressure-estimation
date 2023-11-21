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
import csv
import os
#os.environ["PATH"] += os.pathsep + 'C:/BP vital sign//vitalsign_data/TPR12_yj/'

temp_raw_data = np.load('C://BP vital sign//vitalsign_data//TPR12_yj//re_forehead_human.npy')
time_stamp= pd.read_csv('C://BP vital sign//vitalsign_data//TPR12_yj/time_stamp.csv',header=None)
def read_measurement_elapsed_time(file_name):
    #path = 'C:/BP vital sign/vitalsign_data/TPR11_olfa/'

    f = open('C:/BP vital sign/vitalsign_data/TPR12_yj/time_stamp.csv'.format(file_name), encoding='utf-8')
    rdr = csv.reader(f)

    measure_time = []

    first_time = 0
    for line in rdr:
        if line[0] != 'number' and line[0] != '':
            str_time = line[1]
            hour = int(str_time[-15:-13])
            min = int(str_time[-12:-10])
            sec = int(str_time[-9:-7])
            ms = int(str_time[-6:-1])

            total_time = hour * 60 * 60 * 100000 + min * 60 * 100000 + sec * 100000 + ms

            # if line[0] == '0':
            # first_time = total_time

            if first_time == 0:
                first_time = total_time

            measure_time.append((total_time - first_time) / 100000)

            if measure_time[-1] < 0:
                print("min {} sec {} ms {} idx {}".format(min, sec, ms, line))

    f.close()

    measure_time = np.array(measure_time)

    return measure_time
measure_time = read_measurement_elapsed_time(time_stamp)
#print(measure_time)
#print(len(measure_time))
roi_data = []

for i in range(len(temp_raw_data)):
    temp_data = np.average(np.reshape(temp_raw_data[i][:14, :, :], (14, -1)), axis=1)
    roi_data.append(temp_data)

roi_data = np.array(roi_data)
            
time_start_idx = 0
if len(roi_data) < len(measure_time):
    time_end_idx = int(measure_time[len(roi_data)])-3
else:
    time_end_idx = int(measure_time[-1]) - 3
            
roi_data = np.array(roi_data)
            
start_idx = int(np.where(time_start_idx-1 <= measure_time)[0][0])
end_idx = int(np.where(measure_time <= time_end_idx+1)[0][-1])+10
#interpolation fix
end_idx, _ = np.shape(roi_data)

temp_interpolation = []
set_fps=30
for temp_wave in range(14):
                f_linear_ref = interpolate.interp1d(measure_time[start_idx:end_idx], roi_data[start_idx:end_idx, temp_wave], kind='linear')
                sample_tiem = np.arange(time_start_idx, time_end_idx, (1 / set_fps))
                sample_ref = f_linear_ref(sample_tiem)
                sample_ref = sample_ref[:, np.newaxis]

                if len(temp_interpolation) == 0:
                    temp_interpolation = sample_ref
                else:
                    temp_interpolation = np.concatenate([temp_interpolation, sample_ref], axis=1)
mv_window = 30
temp_interpolation_mv= []
for i in range(len(temp_interpolation)):
    if i < mv_window:
        temp_interpolation_mv.append(temp_interpolation[i])
    else:
        temp_interpolation_mv.append(list(np.average(temp_interpolation[i - mv_window:i, :], axis=0)))
temp_interpolation_mv_data = np.array(temp_interpolation_mv)
            # import matplotlib.pyplot as plt
            # plt.plot(temp_interpolation_mv_data[:,7:8])
absorption_list = []

            
            
def calculate_k(x, y, z):
                # x : 851.35 measure idx : 24
            # y : 490.83 measure idx : 0
            # z : 668.79 measure idx : 10

    a = -54.540945783151464
    b = 21.095479254243322
    c = 78.56709080853545
    d = -4.968144415839242

    t = -(a*x+b*y+c*z+d)/(a+b+c)
    meaure_r_p = x+t
    meaure_g_p = y+t
    meaure_b_p = z+t

    k = x- meaure_r_p

    distance = (((x- meaure_r_p) ** 2) + ((y - meaure_g_p) ** 2) +((z- meaure_b_p) ** 2))/3
    distance = distance**0.5

            # print("Origin point : ", x, ", ",y, ", ",z)
            # print("Covert point : ", meaure_r_p, ", ",meaure_g_p, ", ",meaure_b_p)
            # print("CHECK distance : ", distance, " k : ", k)

    return distance, k
            
k_list = []
for i in range(len(temp_interpolation_mv_data)):
    shading, k = calculate_k(-(np.log(temp_interpolation_mv_data[i][0])),
                                 -(np.log(temp_interpolation_mv_data[i][6])),
                                 -(np.log(temp_interpolation_mv_data[i][13])))

    if i < 30:
        k_list.append(k)

    temp_list = []
    for ii in range(14): # 25
        temp_list.append(-(np.log(temp_interpolation_mv_data[i][ii])) - k)

    absorption_list.append(temp_list)
x, y = np.shape(absorption_list)
print(x, y)
def center(x):
 mean = np.mean(x, axis=0, keepdims=True)
 centered = (x - mean)/mean
 return centered, mean
            
from scipy.signal import butter, lfilter
def butter_bandpass(lowcut, highcut, fs, order=5):
                nyq = 0.5 * fs
                low = lowcut / nyq
                high = highcut / nyq
                b, a = butter(order, [low, high], btype='band')
                return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
                b, a = butter_bandpass(lowcut, highcut, fs, order=order)
                y = lfilter(b, a, data)
                return y
absorption_list=np.array(absorption_list)       
def detrending_componet_fix(data, signal_length):
                x, y = np.shape(data)
                for j in range(0, x):
                    detrending = absorption_list[0+j:signal_length+j, :]
                    ac_com, dc_com = center(detrending)
                    ac_com = np.square(ac_com[0:1,:])
                    # print(np.shape(ac_com[0:1,:]))
                    # dc_com = np.full((signal_length,14), dc_com)
                    if j == 0:
                        detrending_data = ac_com
                        de_DC = dc_com
                    else:
                        detrending_data = np.concatenate(([detrending_data, ac_com]), axis=0)
                        de_DC = np.concatenate(([de_DC, dc_com]), axis=0)
                return detrending_data, de_DC
detrending_data=detrending_componet_fix(absorption_list,x)

detrending_data=np.array(detrending_data)
#print(detrending_data)
#print(shape(detrending_data))
fs = 30.0   
for i in range(0, 14):
                band_data = np.squeeze(detrending_data[:, i:i+1])
                band_data_bp = butter_bandpass_filter(band_data, 0.5, 5, fs)
                if i == 0:
                    detrending_data_bp = band_data_bp
                else:
                    detrending_data_bp = np.vstack(([detrending_data_bp, band_data_bp]))


detrending_data_bp = np.transpose(detrending_data_bp)
print(detrending_data_bp)
#print(type(detrending_data_bp))
#print(np.shape(detrending_data_bp))
detrending_data_bp = detrending_data_bp[30:, :]
#print(np.shape(detrending_data_bp))

#print(detrending_data_bp)
plt.plot(detrending_data_bp)
plt.show()
pd.DataFrame(detrending_data_bp).to_csv("C://BP vital sign//vitalsign_data//TPR12_yj//re_forehead_final pre v2.csv", index=None,line_terminator='\n')

