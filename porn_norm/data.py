#coding:utf-8
"""
Author:hadxu

"""
import os
from PIL import Image
import numpy as np

def load_data():
	data = np.empty((20000,3,32,32),dtype="float32")
	label = np.empty((20000,),dtype="uint8")

	imgs = os.listdir("./imgs_n")
	num = len(imgs)
	for i in range(num):
		img = Image.open("./imgs_n/"+imgs[i])
		arr = np.asarray(img,dtype="float32")
		data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
		label[i] = 0

	imgs = os.listdir("./imgs_p")
	for i in range(num,20000):
		img = Image.open("./imgs_p/"+imgs[i-10000])
		arr = np.asarray(img,dtype="float32")
		data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
		label[i] = 1

	return data,label