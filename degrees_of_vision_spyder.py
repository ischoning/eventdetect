# Create fixation and saccade files and combined data file 
import pandas as pd
import pyreadstat
import numpy as np

from detect.sample import Sample
from detect.sample import ListSampleStream

import os
import math
import statistics as st
import matplotlib.pyplot as plt

from detect.dispersion import *


#xxx = 'C:/Users/marek/source/repos/data-cleanup/preprocessed/new_interpolated/08_interpolated_degrees.csv'
xxx = '1_interpolated_degrees.sav'
degreesTb = pd.read_spss(xxx)
fixationTb = pd.DataFrame()
saccadeTb = pd.DataFrame()

# df = pd.DataFrame(degreesTb)
# print(df.dtypes)
# print(df.iloc[:,:10])
# print(df['raw_timestamp'][1] - df['raw_timestamp'][0])
# print(df['timestamp_milis'][1] - df['timestamp_milis'][0])
# print(df.columns.values)



print('-----------------------Start!-----------------------')  

#print(degreesTb[degreesTb.current_stimulus < 1].index)

if 'old_index' not in fixationTb.columns:
    fixationTb.insert(0,"old_index",np.int)
    
if 'timestamp_milis' not in fixationTb.columns:
    fixationTb.insert(1,"timestamp_milis",np.float32)
    
# centroid of fixation
if 'fixation_x' not in fixationTb.columns:
    fixationTb.insert(2,"fixation_x",0)
    fixationTb['fixation_x'] = fixationTb['fixation_x'].astype(np.float32)
    
# centroid of fixation
if 'fixation_y' not in fixationTb.columns:
    fixationTb.insert(3,"fixation_y",0)
    fixationTb['fixation_y'] = fixationTb['fixation_y'].astype(np.float32)
    
if 'fixation_start_ts' not in fixationTb.columns:
    fixationTb.insert(4,"fixation_start_ts",0)
    fixationTb['fixation_start_ts'] = fixationTb['fixation_start_ts'].astype(np.float32)
    
if 'fixation_end_ts' not in fixationTb.columns:
    fixationTb.insert(5,"fixation_end_ts",0)
    fixationTb['fixation_end_ts'] = fixationTb['fixation_end_ts'].astype(np.float32)
    
if 'saccade_start_ts' not in fixationTb.columns:
    fixationTb.insert(6,"saccade_start_ts",0)
    fixationTb['saccade_start_ts'] = fixationTb['saccade_start_ts'].astype(np.float32)
    
if 'saccade_end_ts' not in fixationTb.columns:
    fixationTb.insert(7,"saccade_end_ts",0)
    fixationTb['saccade_end_ts'] = fixationTb['saccade_end_ts'].astype(np.float32)
    
if 'vision_simulation' not in fixationTb.columns:
    fixationTb.insert(8,"vision_simulation",0)
    
if 'image_enhancement' not in fixationTb.columns:
    fixationTb.insert(9,"image_enhancement",0)
    
if 'current_stimulus' not in fixationTb.columns:
    fixationTb.insert(10,"current_stimulus",0)
    
if 'array_difficulty' not in fixationTb.columns:
    fixationTb.insert(11,"array_difficulty",0)
    
if 'response_correct' not in fixationTb.columns:
    fixationTb.insert(12,"response_correct",0)
    
if 'key_response_time' not in fixationTb.columns:
    fixationTb.insert(13,"key_response_time",np.nan)
    fixationTb['key_response_time'] = fixationTb['key_response_time'].astype(np.float64) 
    
if 'participant_id' not in fixationTb.columns:
    fixationTb.insert(14,"participant_id",0)

if 'saccade_amplitude' not in fixationTb.columns:
    fixationTb.insert(15,"saccade_amplitude",0)
    fixationTb['saccade_amplitude'] = fixationTb['saccade_amplitude'].astype(np.float32)
    
    #sum of saccade_amplitudes per stimulus
if 'scan_path' not in fixationTb.columns:
    fixationTb.insert(16,"scan_path",0)
    fixationTb['scan_path'] = fixationTb['scan_path'].astype(np.float32)
    
if 'saccade_duration' not in fixationTb.columns:
    fixationTb.insert(17,"saccade_duration",0)
    fixationTb['saccade_duration'] = fixationTb['saccade_duration'].astype(np.float32)
#"""  
if 'mean_stimul_fixation' not in fixationTb.columns:
    fixationTb.insert(18,"mean_stimul_fixation",0)
    fixationTb['mean_stimul_fixation'] = fixationTb['mean_stimul_fixation'].astype(np.float32)
    
if 'median_stimul_fixation' not in fixationTb.columns:
    fixationTb.insert(19,"median_stimul_fixation",0)
    fixationTb['median_stimul_fixation'] = fixationTb['median_stimul_fixation'].astype(np.float32)
    
if 'std_stimul_fixation' not in fixationTb.columns:
    fixationTb.insert(20,"std_stimul_fixation",0)
    fixationTb['std_stimul_fixation'] = fixationTb['std_stimul_fixation'].astype(np.float32)
    
if 'mean_stimul_saccade' not in fixationTb.columns:
    fixationTb.insert(21,"mean_stimul_saccade",0)
    fixationTb['mean_stimul_saccade'] = fixationTb['mean_stimul_saccade'].astype(np.float32)
    
if 'median_stimul_saccade' not in fixationTb.columns:
    fixationTb.insert(22,"median_stimul_saccade",0)
    fixationTb['median_stimul_saccade'] = fixationTb['median_stimul_saccade'].astype(np.float32)
    
if 'std_stimul_saccade' not in fixationTb.columns:
    fixationTb.insert(23,"std_stimul_saccade",0)
    fixationTb['std_stimul_saccade'] = fixationTb['std_stimul_saccade'].astype(np.float32)
#"""
# COPY THE KEY TIME RESPONSE ALONG THE WHOLE DATAFRAME
y = 0
for i in range (0, len(degreesTb)):
    if i + y < len(degreesTb.index)-1:
        stimulus = degreesTb.at[i,'current_stimulus']
        krt = degreesTb.at[i,'key_response_time']

        if degreesTb.at[i+y,'current_stimulus'] != 0 :
            ts = degreesTb.at[i,'timestamp_milis']
            ts_offset = degreesTb.at[i,'timestamp_milis']
            
            y = 1
            while degreesTb.at[i+y,'current_stimulus'] == stimulus:
                    degreesTb.at[i + y,'key_response_time'] = krt
                    y += 1
print('-----------------------Done copying!-----------------------')  
"""

if os.path.isfile('C:/Users/marek/source/repos/data-cleanup/preprocessed/new_interpolated/0' +  xxx[54:-5] + '_ts.csv'):
    degreesTb.to_csv('C:/Users/marek/source/repos/data-cleanup/preprocessed/new_interpolated/0' +  xxx[54:-5] + '_ts.csv')
else:
    degreesTb.to_csv('C:/Users/marek/source/repos/data-cleanup/preprocessed/new_interpolated/0' +  xxx[54:-5] + '_ts.csv')
"""    

# KEEP ROW ONLY WITH STIMULUS != 0
#index_zero_stimulus = degreesTb[degreesTb['current_stimulus'] == 0].index
#degreesTb.drop(index_zero_stimulus, inplace=True)      
#print('-----------------------Done removing!----------------------')   

# -----------------------DISPERSION ALGORITHM-----------------------
sampleFields = ['timestamp_milis', 'degrees_right_horizontal', 'degrees_right_vertical']
gazeSamples = []
stdFixations = []
# Store gaze sample data in 'Sample' format, i.e. {index, time, x, y}
for i in degreesTb.index:
    p = Sample(i, degreesTb.at[i, 'timestamp_milis'], degreesTb.at[i, 'new_degrees_RIGHT_horizontal_1'], degreesTb.at[i, 'new_degrees_LEFT_horizontal_1'])
    #p = Sample(i, degreesTb.at[i, 'timestamp_milis'], degreesTb.at[i, 'degrees_right_horizontal'], degreesTb.at[i, 'degrees_right_vertical'])
    gazeSamples.append(p)
    #print('this many')

# Dispersion algorithm:
windowSize = 10 #unit = samples
threshold = 1.5

#initialize Dispersion
stream = ListSampleStream(gazeSamples)
d = Dispersion(stream, windowSize, threshold)
print('before dispersion')
if True:
    Efixation = []

    while True:
        #inputs.append(d.input)
        #windows.append(d.window)
        try:
            dNext = d.next()
            #print(dNext.center.x, dNext.center.y)
            print(dNext)
            Efixation.append(dNext)
            #if (index != 0):
            #    print('dNext',dNext,'dPrev',dPrev)
            #print(dNext.center.x, dNext.center.y, dNext.start.index, dNext.length)
            #print(dNext.center, dNext.start.index, dNext.length)
            #print(dNext)
            #for i in range (0, len(gazeDataTb)):
             #   if gazeDataTb.at[i,'V1'] == 
            #print(dNext)
        except:
            break

print('after dispersion')
print(len(Efixation), Efixation)
# -----------------------COMBINED-----------------------
this_index = 0
print('daco')
for i in Efixation:
    #print(this_index)
    if degreesTb.last_valid_index() > this_index +  i.start.index + i.length + 1:
        fixationTb.at[this_index, 'timestamp_milis'] = degreesTb.at[i.start.index,'timestamp_milis']
        fixationTb.at[this_index, 'fixation_x'] = i.center.x
        fixationTb.at[this_index, 'fixation_y'] = i.center.y
        fixationTb.at[this_index, 'fixation_start_ts'] = degreesTb.at[i.start.index,'timestamp_milis']
        fixationTb.at[this_index, 'fixation_end_ts'] = degreesTb.at[i.start.index + i.length,'timestamp_milis']
    
        fixationTb.at[this_index, 'saccade_start_ts'] = degreesTb.at[i.start.index + i.length + 1 ,'timestamp_milis']
        #this need fixing
        fixationTb.at[this_index, 'saccade_end_ts'] = degreesTb.at[i.start.index - 1 ,'timestamp_milis']
        
        fixationTb.at[this_index, 'vision_simulation'] = degreesTb.at[i.start.index ,'vision_simulation']
        fixationTb.at[this_index, 'image_enhancement'] = degreesTb.at[i.start.index ,'image_enhancement']
        fixationTb.at[this_index, 'current_stimulus'] = degreesTb.at[i.start.index ,'current_stimulus']
        fixationTb.at[this_index, 'array_difficulty'] = degreesTb.at[i.start.index ,'array_difficulty']
        fixationTb.at[this_index, 'response_correct'] = degreesTb.at[i.start.index ,'response_correct']
        fixationTb.at[this_index, 'key_response_time'] = degreesTb.at[i.start.index ,'key_response_time']
        fixationTb.at[this_index, 'old_index'] = i.start.index
        try: fixationTb.at[this_index, 'participant_id'] = int(xxx[54:-25])
        except: fixationTb.at[this_index, 'participant_id'] = None
    
        if not (degreesTb.last_valid_index() - 1) == this_index:
            this_index += 1
        
print('after fixation and saccade calculation')

for i in range (0, len(fixationTb)-1):  # ?????????????????????????????????????????????
    fixationTb.at[i,'saccade_amplitude'] = math.sqrt((fixationTb.at[i+1,'fixation_x'] - 
           fixationTb.at[i,'fixation_x']) *
          (fixationTb.at[i+1,'fixation_x'] -
           fixationTb.at[i,'fixation_x']) +
          (fixationTb.at[i+1,'fixation_y'] -
           fixationTb.at[i,'fixation_y']) *
          (fixationTb.at[i+1,'fixation_y'] -
           fixationTb.at[i,'fixation_y']))
    
    fixationTb.at[i+1,'saccade_duration'] = fixationTb.at[i+1,'fixation_start_ts'] - fixationTb.at[i,'fixation_end_ts']

print('after amplitude and duration calculation')


# Fix the last sample iteration

"""
this_index = 0
this_stimulus = 0
this_fixations = []
this_saccades = []

for i in range (0, len(fixationTb)-1): 
    if i == 0:
        this_stimulus = fixationTb.at[i,'current_stimulus']

    while fixationTb.at[i + this_index,'current_stimulus'] == this_stimulus:
        # ts_last - ts_first
        print(fixationTb.at[i + this_index,'fixation_end_ts'] - fixationTb.at[i + this_index,'fixation_start_ts'])
        temp = fixationTb.at[i + this_index,'fixation_end_ts'] - fixationTb.at[i + this_index,'fixation_start_ts']
        this_fixations.append(float(temp))
        this_saccades.append(xxx.at[i + this_index,'saccade_amplitude'])
        this_index += 1
        #this_stimulus = fixationTb.at[i + this_index,'current_stimulus']
        
    #fixationTb.at[i,'mean_stimul_fixation']
    
    print('mean_stimul_fixation', np.mean(this_fixations))
    print('median_stimul_fixation', np.median(this_fixations))
    print('std_stimul_fixation', np.std(this_fixations))
    
    print('scan_path',np.sum(this_saccades))
    print('mean_stimul_saccade',np.mean(this_saccades))
    print('median_stimul_saccade',np.median(this_saccades))
    print('std_stimul_saccade',np.std(this_saccades))
    
    this_index = 0
    this_stimulus = xxx.at[i+1,'current_stimulus']
print('after mean, meadian and std calculation')


"""

final_data = fixationTb[fixationTb.current_stimulus != 0]
    
if os.path.isfile('/Users/ischoning/PycharmProjects/GitHub/eventdetect/data_out' +  xxx[54:-5] + '_fixations.csv'):
    final_data.to_csv('/Users/ischoning/PycharmProjects/GitHub/eventdetect/data_out' +  xxx[54:-5] + '_fixations.csv')
else:
    #final_data.to_csv('C:/Users/marek/source/repos/data-cleanup/new_procesed/0' +  xxx[54:-5] + '_fixations.csv')
    final_data.to_csv(xxx[54:-5]+'fixations_final_data.csv')
print('-----------------------DISPERSION-----------------------')    


"""
dataTb = pd.DataFrame()
dataTb = pd.merge(fixationTb, saccadeTb, how = "inner", sort = "old_index") 

if 'saccade_amplitude' not in dataTb.columns:
    dataTb.insert(14,"saccade_amplitude",0)
    dataTb['saccade_amplitude'] = dataTb['saccade_amplitude'].astype(np.float32)
    
if 'saccade_duration' not in dataTb.columns:
    dataTb.insert(15,"saccade_duration",0)
    dataTb['saccade_duration'] = dataTb['saccade_duration'].astype(np.float32)
    

# CALCULATE SACCADE AMPLITUES AND DURATIONS
for i in range (0, len(degreesTb)):
    stimulus = degreesTb.at[i,'current_stimulus']
    krt = degreesTb.at[i,'key_response_time']
    

if os.path.isfile('C:/Users/marek/source/repos/data-cleanup/preprocessed/new_interpolated/0'+ xxx[72:-25]  + '_data.csv'):
    dataTb.to_csv('C:/Users/marek/source/repos/data-cleanup/preprocessed/new_interpolated/0' + xxx[72:-25]  + '_data.csv')
else:
    dataTb.to_csv('C:/Users/marek/source/repos/data-cleanup/preprocessed/new_interpolated/0' + xxx[72:-25]  + '_data.csv')
"""
print('-----------------------DONE-----------------------')


# remove all non stimuli samples
# calcluate SACCADE AMPLITUES AND DURATIONS, mean median and all from the paper notes
