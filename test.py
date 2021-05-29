###############################################################################
# Event Detection Algorithm Suite
#  Copyright (C) 2012 Gian Perrone (http://github.com/gian)
#  
#  Permission to use, copy, modify, and distribute this software and its
#  documentation for any purpose and without fee is hereby granted,
#  provided that the above copyright notice appear in all copies and that
#  both the copyright notice and this permission notice and warranty
#  disclaimer appear in supporting documentation, and that the name of
#  the above copyright holders, or their entities, not be used in
#  advertising or publicity pertaining to distribution of the software
#  without specific, written prior permission.
#  
#  The above copyright holders disclaim all warranties with regard to
#  this software, including all implied warranties of merchantability and
#  fitness. In no event shall the above copyright holders be liable for
#  any special, indirect or consequential damages or any damages
#  whatsoever resulting from loss of use, data or profits, whether in an
#  action of contract, negligence or other tortious action, arising out
#  of or in connection with the use or performance of this software.
###############################################################################

from detect.sample import Sample
from detect.sample import ListSampleStream

from detect.dispersion import *
from detect.velocity import *
from detect.hmm import *
from detect.aoi import *
from detect.movingaverage import *
from detect.srr import *
from detect.intersamplevelocity import *
from detect.engbertkliegl import *
from detect.noisefilter import *
from detect.smeetshooge import *

import pandas as pd

def hmm_marek():
	xxx = '1_interpolated_degrees.sav'
	degreesTb = pd.read_spss(xxx)
	fixationTb = pd.DataFrame()
	saccadeTb = pd.DataFrame()

	print('-----------------------Start!-----------------------')

	# print(degreesTb[degreesTb.current_stimulus < 1].index)

	if 'old_index' not in fixationTb.columns:
		fixationTb.insert(0, "old_index", np.int)

	if 'timestamp_milis' not in fixationTb.columns:
		fixationTb.insert(1, "timestamp_milis", np.float32)

	# centroid of fixation
	if 'fixation_x' not in fixationTb.columns:
		fixationTb.insert(2, "fixation_x", 0)
		fixationTb['fixation_x'] = fixationTb['fixation_x'].astype(np.float32)

	# centroid of fixation
	if 'fixation_y' not in fixationTb.columns:
		fixationTb.insert(3, "fixation_y", 0)
		fixationTb['fixation_y'] = fixationTb['fixation_y'].astype(np.float32)

	if 'fixation_start_ts' not in fixationTb.columns:
		fixationTb.insert(4, "fixation_start_ts", 0)
		fixationTb['fixation_start_ts'] = fixationTb['fixation_start_ts'].astype(np.float32)

	if 'fixation_end_ts' not in fixationTb.columns:
		fixationTb.insert(5, "fixation_end_ts", 0)
		fixationTb['fixation_end_ts'] = fixationTb['fixation_end_ts'].astype(np.float32)

	if 'saccade_start_ts' not in fixationTb.columns:
		fixationTb.insert(6, "saccade_start_ts", 0)
		fixationTb['saccade_start_ts'] = fixationTb['saccade_start_ts'].astype(np.float32)

	if 'saccade_end_ts' not in fixationTb.columns:
		fixationTb.insert(7, "saccade_end_ts", 0)
		fixationTb['saccade_end_ts'] = fixationTb['saccade_end_ts'].astype(np.float32)

	if 'vision_simulation' not in fixationTb.columns:
		fixationTb.insert(8, "vision_simulation", 0)

	if 'image_enhancement' not in fixationTb.columns:
		fixationTb.insert(9, "image_enhancement", 0)

	if 'current_stimulus' not in fixationTb.columns:
		fixationTb.insert(10, "current_stimulus", 0)

	if 'array_difficulty' not in fixationTb.columns:
		fixationTb.insert(11, "array_difficulty", 0)

	if 'response_correct' not in fixationTb.columns:
		fixationTb.insert(12, "response_correct", 0)

	if 'key_response_time' not in fixationTb.columns:
		fixationTb.insert(13, "key_response_time", np.nan)
		fixationTb['key_response_time'] = fixationTb['key_response_time'].astype(np.float64)

	if 'participant_id' not in fixationTb.columns:
		fixationTb.insert(14, "participant_id", 0)

	if 'saccade_amplitude' not in fixationTb.columns:
		fixationTb.insert(15, "saccade_amplitude", 0)
		fixationTb['saccade_amplitude'] = fixationTb['saccade_amplitude'].astype(np.float32)

	# sum of saccade_amplitudes per stimulus
	if 'scan_path' not in fixationTb.columns:
		fixationTb.insert(16, "scan_path", 0)
		fixationTb['scan_path'] = fixationTb['scan_path'].astype(np.float32)

	if 'saccade_duration' not in fixationTb.columns:
		fixationTb.insert(17, "saccade_duration", 0)
		fixationTb['saccade_duration'] = fixationTb['saccade_duration'].astype(np.float32)
	# """
	if 'mean_stimul_fixation' not in fixationTb.columns:
		fixationTb.insert(18, "mean_stimul_fixation", 0)
		fixationTb['mean_stimul_fixation'] = fixationTb['mean_stimul_fixation'].astype(np.float32)

	if 'median_stimul_fixation' not in fixationTb.columns:
		fixationTb.insert(19, "median_stimul_fixation", 0)
		fixationTb['median_stimul_fixation'] = fixationTb['median_stimul_fixation'].astype(np.float32)

	if 'std_stimul_fixation' not in fixationTb.columns:
		fixationTb.insert(20, "std_stimul_fixation", 0)
		fixationTb['std_stimul_fixation'] = fixationTb['std_stimul_fixation'].astype(np.float32)

	if 'mean_stimul_saccade' not in fixationTb.columns:
		fixationTb.insert(21, "mean_stimul_saccade", 0)
		fixationTb['mean_stimul_saccade'] = fixationTb['mean_stimul_saccade'].astype(np.float32)

	if 'median_stimul_saccade' not in fixationTb.columns:
		fixationTb.insert(22, "median_stimul_saccade", 0)
		fixationTb['median_stimul_saccade'] = fixationTb['median_stimul_saccade'].astype(np.float32)

	if 'std_stimul_saccade' not in fixationTb.columns:
		fixationTb.insert(23, "std_stimul_saccade", 0)
		fixationTb['std_stimul_saccade'] = fixationTb['std_stimul_saccade'].astype(np.float32)
	# """
	# COPY THE KEY TIME RESPONSE ALONG THE WHOLE DATAFRAME
	y = 0
	for i in range(0, len(degreesTb)):
		if i + y < len(degreesTb.index) - 1:
			stimulus = degreesTb.at[i, 'current_stimulus']
			krt = degreesTb.at[i, 'key_response_time']

			if degreesTb.at[i + y, 'current_stimulus'] != 0:
				ts = degreesTb.at[i, 'timestamp_milis']
				ts_offset = degreesTb.at[i, 'timestamp_milis']

				y = 1
				while degreesTb.at[i + y, 'current_stimulus'] == stimulus:
					degreesTb.at[i + y, 'key_response_time'] = krt
					y += 1
	print('-----------------------Done copying!-----------------------')

	# -----------------------HMM ALGORITHM-----------------------
	sampleFields = ['timestamp_milis', 'degrees_right_horizontal', 'degrees_right_vertical']
	gazeSamples = []
	stdFixations = []
	# Store gaze sample data in 'Sample' format, i.e. {index, time, x, y}
	for i in degreesTb.index:
		p = Sample(i, degreesTb.at[i, 'timestamp_milis'], degreesTb.at[i, 'new_degrees_RIGHT_horizontal_1'],
				   degreesTb.at[i, 'new_degrees_LEFT_horizontal_1'])
		# p = Sample(i, degreesTb.at[i, 'timestamp_milis'], degreesTb.at[i, 'degrees_right_horizontal'], degreesTb.at[i, 'degrees_right_vertical'])
		gazeSamples.append(p)
	# print('this many')

	stream = ListSampleStream(gazeSamples)

	h = HMM(stream, 0.01, 100.0, 4500.0, 100.0, 0.95, 0.05, 0.95, 0.05)
	# TODO: calculate actual distribution values
	for i in h:
		print(i)

def lineto(x1,y1,x2,y2,offset,samples,timeInterval):
	l = []
	t = (float(timeInterval) / float(samples))
	xv = (x2 - x1) / float(samples)
	yv = (y2 - y1) / float(samples)

	p = Sample(offset,timeInterval * offset,x1,y1)
	l.append(p)

	for i in range(0,samples):
		p = Sample(i+offset,timeInterval * (offset + i), p.x+xv, p.y+yv)
		l.append(p)
	
	return l

def saccto(x1,y1,x2,y2,offset,samples,timeInterval):
	""" Construct a fake saccade-like set of samples.
	
	    We work in polar coordinates, because it's easier.

	    We first compute the average velocity for half the displacement:
	    (i.e., to the mid-point)
	    v = 0.5r / t
	    Then acceleration:
	    a = 2v / t
	    Then we can iterate over the time range:
	    d(i) = (a * i) * interval
	"""
	l = []

	p = Sample(offset,timeInterval * offset,x1,y1)
	l.append(p)

	r = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
	theta = math.atan2(y2-y1, x2-x1)

	#print("R: " + str(r) + " theta: " + str(theta))

	vAve = r / float(samples)

	#print("vAve: " + str(vAve))

	a = (2*vAve) / float(samples / 2)

	v = 0
	xo = x1
	yo = y1

	for i in range(1,1 + int(samples / 2)):
		v = v + a

		x = v * math.cos(theta)
		y = v * math.sin(theta)
		xo = xo + x
		yo = yo + y

		#print("V: " + str(v / timeInterval))
	
		p = Sample(i+offset,timeInterval * (offset + i), xo, yo)
		l.append(p)
	
	# Reverse acceleration direction
	a = -a
	offset = offset + (samples / 2) + 1

	for i in range(0,int(samples / 2)-1):
		v = v + a

		x = v * math.cos(theta)
		y = v * math.sin(theta)
		xo = xo + x
		yo = yo + y

		#print("V: " + str(v))
		
		p = Sample(i+offset,timeInterval * (offset + i), xo, yo)
		l.append(p)
	
	return l

# Q: What's offset?
def fixate(x,y,offset,samples,timeInterval):
	l = []

	for i in range(0,samples):
		p = Sample(i+offset,timeInterval * (offset + i), x, y)
		l.append(p)
	
	return l

testPath = fixate(250,250,0,49,0.001)
testPath.extend(saccto(250,250,550,550,50,100,0.001))
testPath.extend(fixate(550,550,151,99,0.001))
testPath.extend(saccto(550,550,350,150,251,99,0.001))
testPath.extend(fixate(350,150,251,49,0.001))

print("============= I-DT test ===============")

stream = ListSampleStream(testPath)
for s in stream:
	print(s)
d = Dispersion(stream, 3, 5)

for i in d:
	print(i)

print("============= I-VT test ===============")
stream = ListSampleStream(testPath)
v = Velocity(IntersampleVelocity(stream), 5)

for i in v:
	print(i)
"""
print("============= I-HMM test ===============")
testPathB = fixate(500,500,0,3,0.001)
testPathB.extend(saccto(500,500,400,400,4,4,0.001))
testPathB.extend(fixate(400,400,9,4,0.001))
testPathB.extend(saccto(400,400,300,300,13,4,0.001))

print(" * Test 1:")

stream = ListSampleStream(testPathB)
h = HMM(stream, 0.01, 100.0, 35355.0, 100.0, 0.95, 0.05, 0.95, 0.05)

for i in h:
	print(i)

print(" * Test 2:")

stream = ListSampleStream(testPath)
h2 = HMM(stream, 0.01, 100.0, 4500.0, 100.0, 0.95, 0.05, 0.95, 0.05)

for i in h2:
	print(i)
"""
print(" * Test 3:")

df = pd.read_csv('/Users/ischoning/PycharmProjects/GitHub/data/participant07_preprocessed172.csv')

# shorten dataset for time efficiency
df = df[0:int(len(df)/100)]

# assign relevant data
lx = df['left_forward_x']
ly = df['left_forward_y']
lz = df['left_forward_z']
rx = df['right_forward_x']
ry = df['right_forward_y']
rz = df['right_forward_z']

# compute angular values
df['Ax_left'] = np.rad2deg(np.arctan2(lx, lz))
df['Ay_left'] = np.rad2deg(np.arctan2(ly, lz))
df['Ax_right'] = np.rad2deg(np.arctan2(rx, rz))
df['Ay_right'] = np.rad2deg(np.arctan2(ry, rz))
df['Avg_angular_x'] = df[['Ax_left', 'Ax_right']].mean(axis = 1)
df['Avg_angular_y'] = df[['Ay_left', 'Ay_right']].mean(axis = 1)

testPathC = []
for i in range(0, len(df)):
	# x1 = df['Avg_angular_x'][i-1]
	# y1 = df['Avg_angular_y'][i-1]
	# x2 = df['Avg_angular_x'][i]
	# y2 = df['Avg_angular_y'][i]
	# isi = df['raw_timestamp'][i] - df['raw_timestamp'][i-1]
	# testPathC.extend(saccto(x1, y1, x2, y2, i, 1, isi))
	p = Sample(i, df['raw_timestamp'][i], df['Avg_angular_x'][i], df['Avg_angular_y'][i])
	testPathC.append(p)

stream = ListSampleStream(testPathC)

h3 = HMM(stream, 0.01, 1.0, 4500.0, 100.0, 0.95, 0.05, 0.95, 0.05)
# TODO: calculate actual distribution values
for i in h3:
	print(i)

#hmm_marek()

#print "============= Prefix test ==============="
#testPathB = fixate(500,500,0,3,0.001)
#testPathB.extend(saccto(500,500,400,400,4,4,0.001))
#testPathB.extend(fixate(400,400,9,4,0.001))
#testPathB.extend(saccto(400,400,300,300,13,4,0.001))
#
#print " * Test 1:"
#
#stream = ListSampleStream(testPathB)
#h = Prefix(stream, 30)
#
#for i in h:
#	print i
#
#print " * Test 2:"
#
#stream = ListSampleStream(testPath)
#h2 = Prefix(stream, 30)
#
#for i in h2:
#	print i
"""
print("============= I-AOI test ===============")
testPathB = fixate(500,500,0,3,0.001)
testPathB.extend(saccto(500,500,400,400,4,4,0.001))
testPathB.extend(fixate(400,400,9,4,0.001))
testPathB.extend(saccto(400,400,300,300,13,4,0.001))

print(" * Test 1:")

stream = ListSampleStream(testPathB)
h = AOI(stream, 3, [(490,490,510,510),(390,390,410,410)])

for i in h:
	print(i)

print("============= MovingAverageFilter test ===============")
testPathB = fixate(500,500,0,4,0.001)
testPathB.extend(saccto(500,500,400,400,5,4,0.001))
testPathB.extend(fixate(400,400,10,4,0.001))
testPathB.extend(saccto(400,400,300,300,14,4,0.001))

print(" * Test 1:")

stream = MovingAverageFilter(ListSampleStream(testPathB),3)
h = AOI(stream, 3, [(490,490,510,510),(390,390,410,410)])

for i in h:
	print(i)

print("============= SRR test ===============")

testPathB = fixate(500,500,0,4,0.001)
testPathB.extend(saccto(500,500,400,400,5,4,0.001))
testPathB.extend(fixate(400,400,10,4,0.001))
testPathB.extend(saccto(400,400,300,300,14,4,0.001))

stream = MovingAverageFilter(NoiseFilter(ListSampleStream(testPathB), 0.2),3)
h = SRR(stream, 12, 100, 1000, 2)

for i in h:
	print(i)

print("============= EngbertKliegl test ============")

testPathB = fixate(500,500,0,4,0.001)
testPathB.extend(saccto(500,500,400,400,5,4,0.001))
testPathB.extend(fixate(400,400,10,4,0.001))
testPathB.extend(saccto(400,400,300,300,14,4,0.001))

stream = NoiseFilter(ListSampleStream(testPathB), 0.1)
h = EngbertKliegl(stream, 10)

for i in h:
	print(i)

print("============= SmeetsHooge test ============")

testPathB = fixate(500,500,0,10,0.001)
testPathB.extend(saccto(500,500,400,400,11,5,0.001))
testPathB.extend(fixate(400,400,17,8,0.001))
testPathB.extend(saccto(400,400,300,300,26,10,0.001))

stream = MovingAverageFilter(ListSampleStream(testPathB),2)
h = SmeetsHooge(stream, 10000, 3, 3)

for i in h:
	print(i)
"""


