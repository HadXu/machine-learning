import glob,os
from PIL import Image
from scipy.misc import imread,imsave,imresize

size = 32,32

folds = os.listdir('data')

print folds


for fold in folds:
	f = os.listdir('data/'+fold)
	for img in f:
		file,ext = os.path.splitext('data/'+fold+'/'+img)
		img = imread('data/'+fold+'/'+img)
		img_tinted = imresize(img,size)
		imsave(file+'.jpg',img_tinted)

