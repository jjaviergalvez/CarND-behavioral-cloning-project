import numpy as np
import matplotlib.pyplot as plt
import cv2

#Preprocessing the data 
def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

def get_red_channel(image_data):
	#take the red channel
	r = image_data[:,:,:,1]
	return np.expand_dims(r, axis=4)

def cut(image_data):
	#cut the image, I considere garbage 
	return image_data[:,80:,:,:]

def resize(image_data):
	n = len(image_data)
	image_data_small = np.empty([n,40,160,1], dtype='float64')
	for i in range(n):
		image = image_data[i,:,:,:]
		image_small = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
		image_data_small[i] =  np.expand_dims(image_small, axis=3)

	return image_data_small

def preprocessing_data(data_in):
	data_out = cut(data_in)
	data_out = get_red_channel(data_out)
	data_out = normalize_grayscale(data_out)
	data_out = resize(data_out)

	return data_out
