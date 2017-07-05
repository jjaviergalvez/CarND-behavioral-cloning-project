import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np
import pickle
import csv

#read the driving_log file
with open('driving_log.csv', 'r') as f:
    _list = list(csv.reader(f))
n = len(_list)

print(n)

#variables to return
X = np.empty([n,160,320,3], dtype='uint8')
y= np.zeros([n])

#fill the variables
for i in range(n):
	X[i] = imread(_list[i][0]).astype(np.int)
	y[i] = _list[i][3]

#arrange into a dict
data = {'features': X, 'labels': y}

#save into *.p file
pickle.dump(data, open( "data.p", "wb" ))