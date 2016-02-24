
# 使用vgg16模型进行图片预测 #  

前面我们学习了使用cifra10来判断图片的类别，今天我们使用更加强大的已经训练好的模型来预测图片的类别，那就是vgg16,对应的供keras使用的模型人家已经帮我们训练好，我可不想卖肾来买一个gpu。。。
对应的模型在 ['vgg16'](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing) 可以下载。估计被墙了，附上链接(http://pan.baidu.com/s/1qX0CJSC)

## 导入必要的库 ##


```python
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
```

    Using Theano backend.
    D:\Anaconda\lib\site-packages\theano-0.8.0.dev0-py2.7.egg\theano\tensor\signal\downsample.py:5: UserWarning: downsample module has been moved to the pool module.
      warnings.warn("downsample module has been moved to the pool module.")
    

## 使用keras建立vgg16模型 ##


```python
def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model
```


```python
model = VGG_16('vgg16_weights.h5')
```


```python
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
```

## 现在我们开始来预测了 ##

首先写一个方法来加载并处理图片


```python
def load_image(imageurl):
    im = cv2.resize(cv2.imread(imageurl),(224,224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im,axis=0)
    return im
```

## 读取vgg16的类别文件 ##


```python
f = open('synset_words.txt','r')
lines = f.readlines()
f.close()
```


```python
def predict(url):
    im = load_image(url)
    pre = np.argmax(model.predict(im))
    print lines[pre]
```


```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib
    


```python
from IPython.display import Image
```


```python
Image('cat1.jpg')
```




![](http://i.imgur.com/z6Fm6S9.jpg)



## 开始预测 ##


```python
predict('cat1.jpg')
```

    n02123045 tabby, tabby cat
    
    


```python
Image('zebra.jpg')
```




![](http://i.imgur.com/jdnky1a.jpg)




```python
predict('zebra.jpg')
```

    n02391049 zebra
    
    


```python
Image('airplane.jpg')
```




![](http://i.imgur.com/FbnCmpC.jpg)




```python
predict('airplane.jpg')
```

    n02690373 airliner
    
    


```python
Image('pig.jpg')
```




![](http://i.imgur.com/vjjU5p4.jpg)




```python
predict('pig.jpg')
```

    n02395406 hog, pig, grunter, squealer, Sus scrofa
    
    

可见，判断率还是很高的。。。。

# 总结 #

通过这次学习，学会了使用keras来搭建模型，使用vgg16这个模型。

