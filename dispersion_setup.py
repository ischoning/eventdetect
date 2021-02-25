# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 18:59:45 2021

@author: marek
"""
# New data for Fiona 17.2.2021
from detect.sample import Sample
from detect.sample import ListSampleStream
from detect.eventstream import EFixation

import os
import math
import statistics as st
import matplotlib.pyplot as plt
import pyreadstat

import numpy as np
from detect.dispersion import *

import pandas as pd

#xxx = 'C:/Users/marek/source/repos/data-cleanup/new_procesed/01_interpolated_degree_fixations.csv'
#fixationsTb = pd.read_csv(xxx)
yyy = '1_interpolated_degrees.sav'
originalTb = pd.read_spss(yyy)

print('Start!')
#"""

#y1 = np.array([], dtype=float) # 'degrees_horizontal'
start_index = np.array([], dtype=float) # 'degrees_vertical'
end_index = np.array([], dtype=float) # 'degrees_vertical'

for i in Efixation:
    start_index = np.append(start_index, i.start.index)
    end_index = np.append(end_index, i.start.index + i.length)
    #fixationsTb.at[i,'old_index']

start_index = start_index.astype(int)
end_index = end_index.astype(int)

#"""

#Size of the positions over time plots
pwidth = 30 
pheight = 4

#Size of the gaze plots
gwidth = 30
gheight = 15

# 'degrees_horizontal', 'degrees_vertical'
leftX = 'degrees_horizontal'
leftY = 'degrees_vertical'

# x = fixationsTb['sample_velocity']
y1 = np.array([], dtype=float) # 'degrees_horizontal'
y2 = np.array([], dtype=float) # 'degrees_vertical'

x = np.array([], dtype=float)



for i in range(0, 500):
    y1 = np.append(y1, [originalTb.at[i,leftX]])
    #print(fixationsTb.at[i+500,leftX])
    y2 = np.append(y2, [originalTb.at[i,leftY]])
    x = np.append(x, [originalTb.at[i,'timestamp_milis']])
    
#print(y1)
#TODO: 
#  - fix the x y over time, proper readable size
#  - coloring the sections of fixations
#  - compare standard deviations
############################################
# plot Left_eye_position_before_cleanup_plot
plt.style.use('classic')
#plt.figure(dpi = 560)
plt.figure(figsize=(pwidth,pheight), dpi = 600)
plt.plot(x, y1, "bx", label=leftX, markersize=1)
plt.plot(x, y2, "rx", label=leftY, markersize=1)

for i in range(0, len(end_index)):

    plt.plot(x[start_index[i]:end_index[i]], y1[start_index[i]:end_index[i]], "c.", markersize=5)
    plt.plot(x[start_index[i]:end_index[i]] , y2[start_index[i]:end_index[i]], "m.", markersize=5)
    
    plt.plot(np.mean(x[start_index[i]:end_index[i]]), np.mean(y1[start_index[i]:end_index[i]]), 'b.', markersize=10, alpha=0.4)
    plt.plot(np.mean(x[start_index[i]:end_index[i]]), np.mean(y2[start_index[i]:end_index[i]]), 'r.', markersize=10, alpha=0.4)

    
plt.ylim(-20,20)
plt.xlim(x[200],x[499])
#ax = plt.axes()
#ax.yaxis.set_major_locator(plt.MultipleLocator(.1))
#ax.xaxis.set_major_locator(plt.MaxNLocator(20))
plt.grid()
#plt.BrokenBarHCollection([0:10], [0:21])
plt.xticks(rotation=10)
plt.xlabel("Time")
plt.ylabel("Coordinates")
plt.legend(loc="upper left")
plt.title("Degrees_of_vision_plot_" + xxx[54:-5])
plt.tight_layout()
plt.savefig("C:/Users/marek/source/repos/data-cleanup/new_procesed/" + xxx[54:-5] + ".png")
plt.show()
print('End!')